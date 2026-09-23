#!/usr/bin/env python3
"""Train k* transfer heads from the final C2 chiN encoders.

This is the C2-compatible replacement for
``mainmodel/scripts/train_single_virtualnode_undirected_freeend_branch_degree_kstar.py``.
The important architectural difference is that C2 uses a positional learnable-node
embedding (80 x 512) and zeroed input node features, rather than the legacy
three-channel handcrafted node features.

The train/validation split is always made from ``n_train_100k`` with random state
42.  The independent seed43 set is loaded only after training and is never used
for checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import re
import sys
from pathlib import Path

# src.plotting imports Matplotlib even though this trainer does not draw figures.
# A local writable cache avoids repeated font-cache construction in parallel jobs.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tapd_mplconfig")

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import trange

PROJECT_ROOT = Path("/home/zxy/project/TAPD")
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_and_process_dataset  # noqa: E402
from src.ira_common import LearnableNodeGINEEncoder, extract_predictions  # noqa: E402
from src.ira_graph_common import SingleTaskRegressor, VirtualNodeWrapper  # noqa: E402
from src.plotting import compute_metrics  # noqa: E402


TRAIN_DATA_DIR = PROJECT_ROOT / "TrainData/data/chiNlimited/n_train_100k"
TEST_DATA_DIR = (
    PROJECT_ROOT
    / "TrainData/data/chiNlimited/n_test_2k_leakage_free_seed43_independent"
)
FINALMODEL_ROOT = PROJECT_ROOT / "baselinemodels/pub0818/gnn/finalmodel"
CHIN_GRAPH_CACHE = (
    PROJECT_ROOT
    / "baselinemodels/pub0818/gnn/hyperparameter_selection_80k/"
    "runs_100k_fixed80_20_validation_only/dataset_cache/n_train_100k_chiN_data.pt"
)
SEEDS = (202601, 202602, 202603, 202604, 202605)
# ``scratch`` deliberately keeps the C2 architecture (including the learnable
# positional node table) but skips loading any chiN checkpoint.  It is the
# true randomly initialized control for the C2 transfer protocols.
TRANSFER_MODES = ("scratch", "frozen_encoder", "finetune_encoder", "two_stage")

TASK_TYPE = "k_wavevector"
HIDDEN_DIM = 512
NUM_LAYERS = 5
DEC_LAYERS = 1
MAX_NODES = 80
EDGE_DIM = 2
EPOCHS = 200
BATCH_SIZE = 32
ENCODER_LR = 5e-4
DECODER_LR = 5e-3
FROZEN_EPOCHS = 50
SPLIT_RANDOM_STATE = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def source_checkpoint(seed: int) -> Path:
    return FINALMODEL_ROOT / f"C2_nested_train__80k__seed{seed}" / "best_model.pth"


def build_model(device: torch.device) -> SingleTaskRegressor:
    encoder = LearnableNodeGINEEncoder(
        max_nodes=MAX_NODES,
        edge_dim=EDGE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    )
    return SingleTaskRegressor(
        encoder,
        hidden_dim=HIDDEN_DIM,
        num_dec_layers=DEC_LAYERS,
    ).to(device)


def load_c2_encoder(
    model: SingleTaskRegressor, checkpoint: Path, device: torch.device
) -> dict[str, object]:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise RuntimeError(f"No encoder.* tensors in {checkpoint}")
    incompatible = model.encoder.load_state_dict(encoder_state, strict=True)
    return {
        "checkpoint": str(checkpoint.resolve()),
        "encoder_tensor_count": len(encoder_state),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def validate_forward(model: SingleTaskRegressor, device: torch.device) -> tuple[int, ...]:
    # Seven ordinary nodes plus the virtual node. The real data wrapper uses the
    # same zero-node/full-edge construction.
    num_nodes = 8
    edges = []
    for node in range(num_nodes - 2):
        edges.extend(((node, node + 1), (node + 1, node)))
    virtual = num_nodes - 1
    for node in range(num_nodes - 1):
        edges.extend(((virtual, node), (node, virtual)))
    graph = Data(
        x=torch.zeros((num_nodes, 3), dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.zeros((len(edges), EDGE_DIM), dtype=torch.float32),
        batch=torch.zeros(num_nodes, dtype=torch.long),
        ptr=torch.tensor([0, num_nodes], dtype=torch.long),
    ).to(device)
    model.eval()
    with torch.no_grad():
        output = model(graph)
    if not torch.isfinite(output).all():
        raise RuntimeError("Non-finite output during checkpoint forward validation")
    return tuple(output.shape)


def derive_kstar_cache_from_chin_graphs(path: Path, cache: Path):
    """Reuse identical graph tensors and replace only the target from sorted YAMLs."""
    dataset = torch.load(CHIN_GRAPH_CACHE, map_location="cpu", weights_only=False)
    yaml_paths = [mat_path.with_suffix(".yml") for mat_path in sorted(path.glob("*.mat"))]
    if len(dataset) != len(yaml_paths):
        raise RuntimeError(
            f"chiN graph cache/YAML count mismatch: {len(dataset)} != {len(yaml_paths)}"
        )
    for graph, yaml_path in zip(dataset, yaml_paths, strict=True):
        legacy_period = None
        with yaml_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("k_star:"):
                    legacy_period = float(stripped.split(":", 1)[1].strip())
                    break
        if legacy_period is None or legacy_period <= 0:
            raise ValueError(f"Missing/non-positive k_star in {yaml_path}")
        wavevector = 2.0 * np.pi / legacy_period
        graph.y = torch.tensor([[np.log10(wavevector)]], dtype=torch.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(cache.suffix + f".tmp.{os.getpid()}")
    torch.save(dataset, temporary)
    os.replace(temporary, cache)
    print(f"Derived k* cache from identical chiN graph tensors: {len(dataset)} samples")
    return dataset


def load_dataset(path: Path, cache: Path | None):
    if cache is not None and cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=False)
    if cache is not None and path == TRAIN_DATA_DIR and CHIN_GRAPH_CACHE.exists():
        return derive_kstar_cache_from_chin_graphs(path, cache)
    dataset = load_and_process_dataset(str(path), task_type=TASK_TYPE)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(cache.suffix + f".tmp.{os.getpid()}")
        torch.save(dataset, temporary)
        os.replace(temporary, cache)
    return dataset


def wrapped(dataset):
    return VirtualNodeWrapper(dataset, node_mode="zero", edge_mode="full", graph_mode="original")


def make_loaders(args, seed: int):
    train_source = load_dataset(TRAIN_DATA_DIR, args.train_cache)
    indices = np.arange(len(train_source))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=SPLIT_RANDOM_STATE
    )
    # Learning-curve runs use paired, nested subsets: for a given seed every
    # protocol sees the same prefix of the same permutation.  The independent
    # seed43 test set remains untouched and is loaded only after model selection.
    if args.stratify_block_count and args.train_size is None:
        raise ValueError("--stratify-block-count requires --train-size")

    block_counts = None
    if args.stratify_block_count:
        mat_paths = sorted(TRAIN_DATA_DIR.glob("*.mat"))
        if len(mat_paths) != len(train_source):
            raise RuntimeError("MAT-file count does not match the cached training dataset")
        block_counts = np.asarray(
            [int(re.match(r"(\d+)blocks_", path.name).group(1)) for path in mat_paths]
        )

    def select_nested(pool, size):
        if not args.stratify_block_count:
            return np.random.default_rng(seed).permutation(pool)[:size]
        per_group = {blocks: size // 4 for blocks in (2, 3, 4, 5)}
        remainder_start = SEEDS.index(seed) % 4
        for offset in range(size % 4):
            per_group[(2, 3, 4, 5)[(remainder_start + offset) % 4]] += 1
        selected = []
        for blocks in (2, 3, 4, 5):
            group = pool[block_counts[pool] == blocks]
            group_size = per_group[blocks]
            if group_size > len(group):
                raise ValueError(f"Requested {group_size} samples from {blocks}-block group")
            selected.append(
                np.random.default_rng(seed + blocks).permutation(group)[:group_size]
            )
        return np.concatenate(selected)

    if args.train_size is not None:
        if not 1 <= args.train_size <= len(train_idx):
            raise ValueError(
                f"--train-size must be in [1, {len(train_idx)}], got {args.train_size}"
            )
        train_idx = select_nested(train_idx, args.train_size)
    if args.validation_size is not None:
        if not 1 <= args.validation_size <= len(val_idx):
            raise ValueError(
                "--validation-size must be in "
                f"[1, {len(val_idx)}], got {args.validation_size}"
            )
        val_idx = select_nested(val_idx, args.validation_size)
    train_dataset = wrapped(Subset(train_source, train_idx))
    val_dataset = wrapped(Subset(train_source, val_idx))
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, train_idx, val_idx


def set_encoder_trainable(model: SingleTaskRegressor, trainable: bool) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = trainable


def optimizer_for(
    model: SingleTaskRegressor, encoder_trainable: bool, encoder_lr: float, decoder_lr: float
):
    if encoder_trainable:
        groups = [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.decoder.parameters(), "lr": decoder_lr},
        ]
    else:
        groups = [{"params": model.decoder.parameters(), "lr": decoder_lr}]
    return torch.optim.Adam(groups)


def train_phase(
    model,
    train_loader,
    val_loader,
    device,
    optimizer,
    start_epoch,
    epochs,
    encoder_trainable,
    best_val,
    checkpoint,
    history,
):
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    for epoch in trange(start_epoch, start_epoch + epochs, desc="Training", unit="epoch"):
        model.train()
        # Freezing parameters alone still updates BatchNorm buffers. Keeping the
        # encoder in eval mode makes frozen_encoder and stage 1 genuinely frozen.
        if not encoder_trainable:
            model.encoder.eval()
        train_total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = criterion(prediction, batch.y)
            loss.backward()
            optimizer.step()
            train_total += float(loss.item()) * int(batch.num_graphs)
        train_loss = train_total / len(train_loader.dataset)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss = criterion(model(batch), batch.y)
                val_total += float(loss.item()) * int(batch.num_graphs)
        val_loss = val_total / len(val_loader.dataset)
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": int(epoch),
                "phase": "joint" if encoder_trainable else "frozen",
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint)
    return best_val


def save_history(history: list[dict[str, object]], output_dir: Path) -> None:
    write_json(output_dir / "loss_history.json", history)
    with (output_dir / "loss_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["epoch", "phase", "train_loss", "val_loss", "learning_rates"]
        )
        writer.writeheader()
        for row in history:
            serial = dict(row)
            serial["learning_rates"] = ";".join(map(str, row["learning_rates"]))
            writer.writerow(serial)


def evaluate_seed43(model, args, device, output_dir):
    test_source = load_dataset(TEST_DATA_DIR, args.test_cache)
    test_loader = DataLoader(
        wrapped(test_source),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    y_pred, y_true = extract_predictions(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    ape = np.abs(y_pred - y_true) / np.clip(np.abs(y_true), 1e-12, None) * 100.0
    np.savez(
        output_dir / "seed43_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        abs_error=np.abs(y_pred - y_true),
        ape_percent=ape,
    )
    result = {
        "test_data_dir": str(TEST_DATA_DIR),
        "test_size": len(test_source),
        "MAPE": float(metrics["MAPE"]),
        "RMSE": float(metrics["RMSE"]),
        "MAE": float(metrics["MAE"]),
        "R2": float(metrics["R²"]),
        "MaxAE": float(metrics["MaxAE"]),
    }
    write_json(output_dir / "seed43_metrics.json", result)
    return result


def run(args) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    model = build_model(device)
    checkpoint = None
    provenance = None
    if args.transfer_mode != "scratch":
        checkpoint = source_checkpoint(args.seed)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        provenance = load_c2_encoder(model, checkpoint, device)
    forward_shape = validate_forward(model, device)
    if args.transfer_mode == "scratch":
        print(
            f"Initialized C2 model randomly with seed={args.seed}; "
            f"forward shape={forward_shape}"
        )
    else:
        print(
            f"Validated {checkpoint.name}: {provenance['encoder_tensor_count']} encoder tensors, "
            f"forward shape={forward_shape}"
        )
    if args.validate_checkpoint_only:
        return
    if args.prepare_data_cache_only:
        if args.train_cache is None or args.test_cache is None:
            raise ValueError("--prepare-data-cache-only requires --train-cache and --test-cache")
        train_source = load_dataset(TRAIN_DATA_DIR, args.train_cache)
        test_source = load_dataset(TEST_DATA_DIR, args.test_cache)
        print(f"Prepared caches: train={len(train_source)}, seed43_test={len(test_source)}")
        return

    if args.output_dir.exists():
        if any(args.output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output_dir}")
    else:
        args.output_dir.mkdir(parents=True)

    train_loader, val_loader, train_idx, val_idx = make_loaders(args, args.seed)
    np.savez(args.output_dir / "split_indices.npz", train_idx=train_idx, val_idx=val_idx)
    history: list[dict[str, object]] = []
    best_val = float("inf")
    best_path = args.output_dir / "best_model.pth"

    if args.transfer_mode == "scratch":
        # Keep the C2 transfer learning-rate convention for a fair control:
        # encoder 5e-4 and decoder 5e-3 from epoch 1, with no pretrained state.
        set_encoder_trainable(model, True)
        optimizer = optimizer_for(model, True, args.encoder_lr, args.decoder_lr)
        best_val = train_phase(
            model, train_loader, val_loader, device, optimizer, 1, args.epochs,
            True, best_val, best_path, history,
        )
    elif args.transfer_mode == "frozen_encoder":
        set_encoder_trainable(model, False)
        optimizer = optimizer_for(model, False, args.encoder_lr, args.decoder_lr)
        best_val = train_phase(
            model, train_loader, val_loader, device, optimizer, 1, args.epochs,
            False, best_val, best_path, history,
        )
    elif args.transfer_mode == "finetune_encoder":
        set_encoder_trainable(model, True)
        optimizer = optimizer_for(model, True, args.encoder_lr, args.decoder_lr)
        best_val = train_phase(
            model, train_loader, val_loader, device, optimizer, 1, args.epochs,
            True, best_val, best_path, history,
        )
    else:
        frozen_epochs = min(args.frozen_epochs, args.epochs)
        set_encoder_trainable(model, False)
        optimizer = optimizer_for(model, False, args.encoder_lr, args.decoder_lr)
        best_val = train_phase(
            model, train_loader, val_loader, device, optimizer, 1, frozen_epochs,
            False, best_val, best_path, history,
        )
        remaining = args.epochs - frozen_epochs
        if remaining:
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
            set_encoder_trainable(model, True)
            optimizer = optimizer_for(model, True, args.encoder_lr, args.decoder_lr)
            best_val = train_phase(
                model, train_loader, val_loader, device, optimizer,
                frozen_epochs + 1, remaining, True, best_val, best_path, history,
            )

    save_history(history, args.output_dir)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_metrics = evaluate_seed43(model, args, device, args.output_dir)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = {
        "experiment": "C2_learnable_node_KStar_transfer",
        "transfer_mode": args.transfer_mode,
        "seed": args.seed,
        "source_checkpoint": provenance,
        "architecture": {
            "encoder": "LearnableNodeGINEEncoder",
            "max_nodes": MAX_NODES,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "decoder_layers": DEC_LAYERS,
            "edge_features": ["f", "A/B type"],
            "node_features": "learnable positional node embedding; data.x zeroed",
            "virtual_node": True,
            "initialization": (
                "random seeded initialization; no pretrained encoder"
                if args.transfer_mode == "scratch"
                else "C2 chiN checkpoint encoder"
            ),
            "parameter_count": parameter_count,
        },
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "encoder_lr": args.encoder_lr,
            "decoder_lr": args.decoder_lr,
            "frozen_epochs": args.frozen_epochs if args.transfer_mode == "two_stage" else 0,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=10)",
            "best_validation_loss": best_val,
        },
        "data": {
            "task_type": TASK_TYPE,
            "train_data_dir": str(TRAIN_DATA_DIR),
            "test_data_dir": str(TEST_DATA_DIR),
            "train_size": len(train_idx),
            "validation_size": len(val_idx),
            "learning_curve_train_size_requested": args.train_size,
            "learning_curve_validation_size_requested": args.validation_size,
            "subset_protocol": (
                "seed-paired nested, exactly balanced 2/3/4/5-block subsets of "
                "the fixed random-state-42 train/validation split"
                if args.stratify_block_count
                else "seed-paired nested prefix of a seeded permutation of the fixed "
                "random-state-42 train/validation split"
            ),
            "stratified_by_block_count": args.stratify_block_count,
            "split_random_state": SPLIT_RANDOM_STATE,
            "test_role": "independent final inference only; never used for selection",
        },
        "runtime": {
            "device": str(device),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "seed43_metrics": test_metrics,
    }
    write_json(args.output_dir / "run_config.json", config)
    write_json(args.output_dir / "metrics.json", test_metrics)
    print(json.dumps(test_metrics, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-mode", required=True, choices=TRANSFER_MODES)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--encoder-lr", type=float, default=ENCODER_LR)
    parser.add_argument("--decoder-lr", type=float, default=DECODER_LR)
    parser.add_argument("--frozen-epochs", type=int, default=FROZEN_EPOCHS)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--train-size",
        type=int,
        help="Use this many examples from the fixed 80k training pool.",
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        help="Use this many examples from the fixed 20k validation pool.",
    )
    parser.add_argument("--stratify-block-count", action="store_true")
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--test-cache", type=Path)
    parser.add_argument("--validate-checkpoint-only", action="store_true")
    parser.add_argument("--prepare-data-cache-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
