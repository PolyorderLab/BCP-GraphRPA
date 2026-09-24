#!/usr/bin/env python3
"""Train PolymerType models using only a fixed train/validation development split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(__file__).resolve().parent / "dependencies_handoff/models"
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(ROOT))

from baseline_common import (  # noqa: E402
    evaluate_loss,
    load_yml_records,
    predict,
    set_seed,
    train_epoch,
)
from train_transformer_polymer_type import (  # noqa: E402
    PolymerTypeDataset,
    PolymerTypeTransformerRegressor,
    forward_batch,
    polymer_type_collate,
)
from src.plotting import compute_metrics  # noqa: E402


SPLIT_SEED = 42
VAL_FRACTION = 0.2


def write_predictions(path, names, y_true, y_pred):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["base_name", "true", "pred"])
        writer.writerows(zip(names, y_true.tolist(), y_pred.tolist()))


def build_model(args, train_dataset):
    return PolymerTypeTransformerRegressor(
                vocab_size=len(train_dataset.vocab),
                d_model=args.d_model,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dim_feedforward=args.d_model * args.ff_mult,
                dropout=args.dropout,
                max_len=train_dataset.max_len,
                readout=args.readout,
                position_encoding=args.position_encoding,
                numeric_mode="side_channel",
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--records-cache")
    parser.add_argument(
        "--split-indices",
        help=(
            "Optional NPZ containing train_indices and validation_indices into the "
            "records cache. This is used for fixed-common-validation and nested-pool studies."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=0,
        help="Do not allow early stopping before this many epochs have completed.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--readout", choices=["cls", "mean", "cls_mean", "attention"], default="cls")
    parser.add_argument("--position-encoding", choices=["learnable", "sinusoidal", "none"], default="learnable")
    parser.add_argument("--representation", choices=["full_topology", "no_shared_junction", "species_fraction_only"], default="full_topology")
    parser.add_argument("--architecture", choices=["transformer"], default="transformer")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_epochs < 0:
        raise ValueError("--min-epochs must be non-negative")
    if args.min_epochs > args.epochs:
        raise ValueError("--min-epochs cannot exceed --epochs")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.records_cache and args.max_samples is None:
        cache_path = Path(args.records_cache)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Records cache is required for formal runs but does not exist: {cache_path}"
            )
        with cache_path.open("rb") as handle:
            cache = pickle.load(handle)
        if str(Path(args.data_dir).resolve()) != cache["data_directory"]:
            raise ValueError("Records cache data-directory mismatch")
        records = cache["records"]
    else:
        records = load_yml_records(args.data_dir, "chiN", max_samples=args.max_samples)
    if args.split_indices:
        split_path = Path(args.split_indices)
        if not split_path.exists():
            raise FileNotFoundError(f"Split-index file does not exist: {split_path}")
        with np.load(split_path) as split:
            if "train_indices" not in split or "validation_indices" not in split:
                raise ValueError(
                    f"{split_path} must contain train_indices and validation_indices"
                )
            train_indices = np.asarray(split["train_indices"], dtype=np.int64)
            val_indices = np.asarray(split["validation_indices"], dtype=np.int64)
        if len(train_indices) == 0 or len(val_indices) == 0:
            raise ValueError(f"Empty train or validation indices in {split_path}")
        combined = np.concatenate([train_indices, val_indices])
        if combined.min() < 0 or combined.max() >= len(records):
            raise ValueError(f"Out-of-bounds index in {split_path}")
        if len(np.unique(train_indices)) != len(train_indices):
            raise ValueError(f"Duplicate train index in {split_path}")
        if len(np.unique(val_indices)) != len(val_indices):
            raise ValueError(f"Duplicate validation index in {split_path}")
        if np.intersect1d(train_indices, val_indices).size:
            raise ValueError(f"Train/validation overlap in {split_path}")
    else:
        all_indices = np.arange(len(records))
        train_indices, val_indices = train_test_split(
            all_indices,
            test_size=VAL_FRACTION,
            random_state=SPLIT_SEED,
        )
    train_records = [records[int(index)] for index in train_indices]
    val_records = [records[int(index)] for index in val_indices]

    # Fit vocabulary and maximum sequence length on fitting records only.
    train_dataset = PolymerTypeDataset(
        train_records,
        order_variant="canonical",
        seed=args.seed,
        max_len=128,
        numeric_mode="side_channel",
        representation=args.representation,
        canonicalization="strict",
    )
    val_dataset = PolymerTypeDataset(
        val_records,
        vocab=train_dataset.vocab,
        order_variant="canonical",
        seed=args.seed,
        max_len=train_dataset.max_len,
        numeric_mode="side_channel",
        representation=args.representation,
        canonicalization="strict",
    )

    shuffle_generator = torch.Generator().manual_seed(args.seed)
    drop_last = len(train_dataset) > args.batch_size and len(train_dataset) % args.batch_size == 1
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=shuffle_generator,
        collate_fn=polymer_type_collate,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=polymer_type_collate,
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    model = build_model(args, train_dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    criterion = nn.MSELoss()
    checkpoint_path = output_dir / "best_model.pth"
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    print(f"Run: {args.run_name}", flush=True)
    print(f"Train/validation: {len(train_dataset)}/{len(val_dataset)}", flush=True)
    print("Final-test data loaded: false", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, forward_batch)
        val_loss = evaluate_loss(model, val_loader, criterion, device, forward_batch)
        scheduler.step(val_loss)
        history.append((epoch, train_loss, val_loss, optimizer.param_groups[0]["lr"]))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}: train={train_loss:.8f} val={val_loss:.8f} "
                f"lr={optimizer.param_groups[0]['lr']:.3g}",
                flush=True,
            )
        if epoch >= args.min_epochs and stale_epochs >= args.early_stopping_patience:
            print(f"Early stop: epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
    val_pred, val_true, val_names = predict(model, val_loader, device, forward_batch)
    val_metrics = compute_metrics(val_true, val_pred)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    result = {
        "run_name": args.run_name,
        "seed": args.seed,
        "train_count": len(train_dataset),
        "validation_count": len(val_dataset),
        "test_loaded": False,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_val_loss": float(best_val_loss),
        "validation_MAPE": float(val_metrics["MAPE"]),
        "validation_RMSE": float(val_metrics["RMSE"]),
        "validation_MAE": float(val_metrics["MAE"]),
        "validation_R2": float(val_metrics["R²"]),
        "validation_MaxAE": float(val_metrics["MaxAE"]),
        "trainable_parameters": trainable_parameters,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "vocab.json").write_text(
        json.dumps(train_dataset.vocab, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "development_directory": str(Path(args.data_dir).resolve()),
        "split_seed": SPLIT_SEED,
        "validation_fraction": VAL_FRACTION,
        "train_count": len(train_records),
        "validation_count": len(val_records),
        "test_directory": None,
        "test_loaded": False,
        "split_indices_file": str(Path(args.split_indices).resolve()) if args.split_indices else None,
        "train_base_names": [record["base_name"] for record in train_records],
        "validation_base_names": [record["base_name"] for record in val_records],
    }
    (output_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_log_mse", "validation_log_mse", "learning_rate"])
        writer.writerows(history)
    write_predictions(output_dir / "validation_predictions.csv", val_names, val_true, val_pred)
    snapshot_dir = output_dir / "source_snapshot"
    snapshot_dir.mkdir(exist_ok=True)
    shutil.copy2(__file__, snapshot_dir / Path(__file__).name)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
