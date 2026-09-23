import argparse
import csv
import json
import os
import platform
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import mpltex

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PACKAGE_ROOT))
from src.data_utils import load_and_process_dataset
from src.ira_graph_common import (
    BATCH_SIZE,
    DEC_LAYERS,
    EPOCHS,
    HIDDEN_DIM,
    LEARNING_RATE,
    NUM_LAYERS,
    TEST_DATA_DIR,
    TRAIN_DATA_DIR,
    GINEEncoder,
    SingleTaskRegressor,
    VirtualNodeWrapper,
    extract_predictions,
)
from src.plotting import compute_metrics
from src.train_utils import Trainer


EXPERIMENT = "VN_FreeBranchDegree"
IN_FEATS = 3
EDGE_DIM = 2
DEFAULT_RESULT_ROOT = os.path.join(
    str(PACKAGE_ROOT / "results" / "training_runs"),
    f"{EXPERIMENT}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
NOTE = (
    "Virtual node with original bidirectional graph representation. Node features are "
    "[is_free_end, is_branch_point, degree], where all three channels are "
    "derived from the original degree column. Edge features are [f, A/B type]. "
    "The original bidirectional topology and edge attributes are kept unchanged; "
    "a virtual node is added and connected bidirectionally to all nodes."
)


class FreeendBranchDegreeNodeFeatureWrapper:
    """Rebuild node features as [is_free_end, is_branch_point, degree]."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx].clone()
        functionality = data.x[:, 0:1].clone()
        is_free_end = (functionality == 1).to(data.x.dtype)
        is_branch_point = (functionality > 1).to(data.x.dtype)
        degree = functionality.clone()
        data.x = torch.cat([is_free_end, is_branch_point, degree], dim=1)
        return data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total": total, "trainable": trainable}


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_loss_history(train_hist, val_hist, result_root):
    rows = []
    for epoch, (train_loss, val_loss) in enumerate(zip(train_hist, val_hist), start=1):
        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )

    csv_path = os.path.join(result_root, "loss_history.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(rows)

    write_json(os.path.join(result_root, "loss_history.json"), rows)


def read_loss_history(result_root):
    loss_path = os.path.join(result_root, "loss_history.csv")
    train_hist = []
    val_hist = []
    with open(loss_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_hist.append(float(row["train_loss"]))
            val_hist.append(float(row["val_loss"]))
    return train_hist, val_hist


@mpltex.acs_decorator
def save_loss_curve_mpltex(train_hist, val_hist, save_prefix):
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.linewidth"] = 1.2
    fig, ax = plt.subplots(figsize=(7, 5))
    epochs = np.arange(len(train_hist))
    ax.plot(epochs, train_hist, label="Train Loss", color="royalblue", linewidth=2.0)
    ax.plot(epochs, val_hist, label="Val Loss", color="orange", linewidth=2.0)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.set_yscale("log")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    ax.tick_params(axis="both", which="both", direction="out", width=1.1, length=4, top=False, right=False)
    fig.tight_layout()
    fig.savefig(save_prefix + ".png", dpi=300)
    fig.savefig(save_prefix + ".svg")
    plt.close(fig)


@mpltex.acs_decorator
def save_parity_plot_mpltex(y_true, y_pred, save_prefix, title, metrics):
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.linewidth"] = 1.4
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        y_true,
        y_pred,
        c="#0F4D92",
        alpha=0.6,
        edgecolors="white",
        linewidths=0.5,
        s=40,
        rasterized=True,
    )
    lims = [
        min(float(y_true.min()), float(y_pred.min())) * 0.8,
        max(float(y_true.max()), float(y_pred.max())) * 1.2,
    ]
    ax.plot(lims, lims, "--", color="0.25", alpha=0.9, linewidth=2.0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"True $\chi N^*$", fontsize=14)
    ax.set_ylabel(r"Predicted $\chi N^*$", fontsize=14)
    ax.set_title(r"$\chi N^*$", fontsize=16, pad=10)
    stats_text = (
        f"$R^2$ = {metrics['R²']:.4f}\n"
        f"$MAPE$ = {metrics['MAPE']:.2f}%\n"
        f"$RMSE$ = {metrics['RMSE']:.2f}"
    )
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.55", linewidth=1.2, alpha=0.92),
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)
    ax.tick_params(direction="out", width=1.2, length=5, labelsize=11)
    fig.tight_layout()
    fig.savefig(save_prefix + ".png", dpi=300)
    fig.savefig(save_prefix + ".svg")
    plt.close(fig)


def save_predictions(y_true, y_pred, result_root):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    abs_error = np.abs(y_pred - y_true)
    denom = np.clip(np.abs(y_true), 1e-12, None)
    ape = abs_error / denom * 100.0

    csv_path = os.path.join(result_root, "test_predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_index", "y_true", "y_pred", "abs_error", "ape_percent"],
        )
        writer.writeheader()
        for idx, values in enumerate(zip(y_true, y_pred, abs_error, ape)):
            true_value, pred_value, err_value, ape_value = values
            writer.writerow(
                {
                    "sample_index": idx,
                    "y_true": float(true_value),
                    "y_pred": float(pred_value),
                    "abs_error": float(err_value),
                    "ape_percent": float(ape_value),
                }
            )

    np.savez(
        os.path.join(result_root, "test_predictions.npz"),
        y_true=y_true,
        y_pred=y_pred,
        abs_error=abs_error,
        ape_percent=ape,
    )


def save_model_artifacts(model, result_root):
    parameter_counts = count_parameters(model)
    model_info = {
        "model_class": model.__class__.__name__,
        "parameter_counts": parameter_counts,
        "model_repr": repr(model),
    }
    write_json(os.path.join(result_root, "model_summary.json"), model_info)

    with open(os.path.join(result_root, "model_structure.txt"), "w", encoding="utf-8") as f:
        f.write(repr(model))
        f.write("\n\n")
        f.write(f"Total parameters: {parameter_counts['total']}\n")
        f.write(f"Trainable parameters: {parameter_counts['trainable']}\n")


def save_run_config(result_root, seed, run_id, device, split_info):
    config = {
        "experiment": EXPERIMENT,
        "run_id": run_id,
        "seed": seed,
        "note": NOTE,
        "paths": {
            "train_data_dir": TRAIN_DATA_DIR,
            "test_data_dir": TEST_DATA_DIR,
            "result_root": result_root,
        },
        "features": {
            "node_features": ["is_free_end", "is_branch_point", "degree"],
            "edge_features": ["f", "A/B type"],
            "virtual_node": True,
            "graph_topology": "original_bidirectional",
            "undirected_transform": None,
        },
        "hyperparameters": {
            "hidden_dim": HIDDEN_DIM,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "num_layers": NUM_LAYERS,
            "dec_layers": DEC_LAYERS,
            "epochs": EPOCHS,
            "in_feats": IN_FEATS,
            "edge_dim": EDGE_DIM,
        },
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "split": split_info,
    }
    write_json(os.path.join(result_root, "run_config.json"), config)


def load_wrapped_splits():
    full_train_dataset = load_and_process_dataset(TRAIN_DATA_DIR, task_type="chiN")
    test_dataset = load_and_process_dataset(TEST_DATA_DIR, task_type="chiN")
    train_idx, val_idx = train_test_split(
        np.arange(len(full_train_dataset)),
        test_size=0.2,
        random_state=42,
    )
    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(full_train_dataset, val_idx)
    split_info = {
        "split_random_state": 42,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_dataset)),
    }
    return (
        VirtualNodeWrapper(
            FreeendBranchDegreeNodeFeatureWrapper(train_subset),
            node_mode="original",
        ),
        VirtualNodeWrapper(
            FreeendBranchDegreeNodeFeatureWrapper(val_subset),
            node_mode="original",
        ),
        VirtualNodeWrapper(
            FreeendBranchDegreeNodeFeatureWrapper(test_dataset),
            node_mode="original",
        ),
        split_info,
        train_idx,
        val_idx,
    )


def build_model(device):
    encoder = GINEEncoder(
        in_feats=IN_FEATS,
        edge_dim=EDGE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    )
    return SingleTaskRegressor(encoder, HIDDEN_DIM, num_dec_layers=DEC_LAYERS).to(device)


def save_evaluation_outputs(model, test_loader, device, result_root, seed, run_id):
    model_path = os.path.join(result_root, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    y_pred, y_true = extract_predictions(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    save_predictions(y_true, y_pred, result_root)

    save_parity_plot_mpltex(
        y_true,
        y_pred,
        os.path.join(result_root, "parity_plot_chiN"),
        title=f"{EXPERIMENT} seed {seed}",
        metrics=metrics,
    )

    result = {
        "experiment": EXPERIMENT,
        "run_id": run_id,
        "seed": seed,
        "MAPE": float(metrics["MAPE"]),
        "RMSE": float(metrics["RMSE"]),
        "MAE": float(metrics["MAE"]),
        "R2": float(metrics["R²"]),
        "MaxAE": float(metrics["MaxAE"]),
        "result_dir": result_root,
    }
    with open(os.path.join(result_root, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    report = (
        f"chiN single-run experiment: {EXPERIMENT}\n"
        f"{'='*45}\n"
        f"Run: {run_id if run_id is not None else 'single'}\n"
        f"Seed: {seed}\n"
        f"Hyperparameters:\n"
        f"  - hidden_dim: {HIDDEN_DIM}\n"
        f"  - learning_rate: {LEARNING_RATE}\n"
        f"  - batch_size: {BATCH_SIZE}\n"
        f"  - num_layers: {NUM_LAYERS}, dec_layers: {DEC_LAYERS}\n"
        f"  - epochs: {EPOCHS}\n"
        f"{'='*45}\n"
        f"Test Set Results:\n"
        f"  - MAPE : {metrics['MAPE']:.2f}%\n"
        f"  - RMSE : {metrics['RMSE']:.2f}\n"
        f"  - MAE  : {metrics['MAE']:.2f}\n"
        f"  - R2   : {metrics['R²']:.4f}\n"
        f"  - MaxAE: {metrics['MaxAE']:.2f}\n"
        f"{'='*45}\n"
        f"{NOTE}\n"
    )
    with open(os.path.join(result_root, "evaluation_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    return result


def eval_only(result_root, seed=2026, run_id=None):
    set_seed(seed)
    if not os.path.exists(os.path.join(result_root, "best_model.pth")):
        raise FileNotFoundError(f"Missing best_model.pth under {result_root}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, test_dataset, split_info, train_idx, val_idx = load_wrapped_splits()
    del train_dataset, val_dataset
    np.savez(
        os.path.join(result_root, "split_indices.npz"),
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = build_model(device)
    save_model_artifacts(model, result_root)
    save_run_config(result_root, seed, run_id, device, split_info)

    loss_path = os.path.join(result_root, "loss_history.csv")
    if os.path.exists(loss_path):
        train_hist, val_hist = read_loss_history(result_root)
        save_loss_curve_mpltex(train_hist, val_hist, os.path.join(result_root, "loss_curve"))

    return save_evaluation_outputs(model, test_loader, device, result_root, seed, run_id)


def train_one(result_root, seed=2026, run_id=None):
    set_seed(seed)
    os.makedirs(result_root, exist_ok=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, test_dataset, split_info, train_idx, val_idx = load_wrapped_splits()
    np.savez(
        os.path.join(result_root, "split_indices.npz"),
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = build_model(device)
    save_model_artifacts(model, result_root)
    save_run_config(result_root, seed, run_id, device, split_info)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )
    criterion = nn.MSELoss()

    model_path = os.path.join(result_root, "best_model.pth")
    trainer = Trainer(model, optimizer, criterion, device, scheduler=scheduler, save_path=model_path)
    train_hist, val_hist = trainer.fit(train_loader, val_loader, epochs=EPOCHS)
    save_loss_history(train_hist, val_hist, result_root)
    save_loss_curve_mpltex(train_hist, val_hist, os.path.join(result_root, "loss_curve"))

    return save_evaluation_outputs(model, test_loader, device, result_root, seed, run_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.eval_only:
        eval_only(args.result_root, seed=args.seed, run_id=args.run_id)
    else:
        train_one(args.result_root, seed=args.seed, run_id=args.run_id)


if __name__ == "__main__":
    main()
