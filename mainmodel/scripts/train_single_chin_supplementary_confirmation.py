#!/usr/bin/env python3
"""Single-run trainer for supplementary chiN confirmation experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
# External data are deliberately not duplicated in this reproducibility bundle.
# Set TAPD_DATA_ROOT to the directory that contains ``chiNlimited/``.
PROJECT_ROOT = Path(os.environ.get("TAPD_PROJECT_ROOT", PACKAGE_ROOT))
DATA_ROOT = Path(os.environ.get("TAPD_DATA_ROOT", PACKAGE_ROOT / "data"))
sys.path.append(str(PROJECT_ROOT))

from mainmodel.scripts import train_single_virtualnode_undirected_freeend_branch_degree as base
from src.data_utils import load_and_process_dataset
from src.ira_common import LearnableNodeGINEEncoder, LineGraphWrapper, extract_predictions
from src.ira_graph_common import (
    BipartiteGraphWrapper,
    DirectedGraphWrapper,
    GINEEncoder,
    SelfLoopGraphWrapper,
    SingleTaskRegressor,
    VirtualNodeWrapper,
)
from src.plotting import compute_metrics
from src.train_utils import Trainer


TRAIN_DATA_DIR = os.environ.get(
    "TAPD_TRAIN_DATA_DIR", str(DATA_ROOT / "chiNlimited/n_train_100k")
)
TEST_DATA_DIR = os.environ.get(
    "TAPD_VALIDATION_DATA_DIR", str(DATA_ROOT / "chiNlimited/n_test_2k_leakage_free_seed42")
)
SEEDS = [202601, 202602, 202603, 202604, 202605]
MAX_NODES_LEARNABLE = 80
DEFAULT_BATCH_ROOT = (
    PACKAGE_ROOT / "results" / "training_runs" / "chiN_final_confirmation_5runs"
)

COMMON_CONFIG = {
    "task": "single-task chiN",
    "train_data_dir": TRAIN_DATA_DIR,
    "test_data_dir": TEST_DATA_DIR,
    "epochs": 200,
    "hidden_dim": 256,
    "learning_rate": 0.005,
    "batch_size": 32,
    "num_layers": 5,
    "dec_layers": 1,
    "val_fraction": 0.2,
    "split_random_state": 42,
    "optimizer": "Adam",
    "scheduler": {
        "name": "ReduceLROnPlateau",
        "mode": "min",
        "factor": 0.5,
        "patience": 10,
    },
    "loss": "MSE on log10(chiN*)",
    "eval": "convert back to original scale; report MAPE, RMSE, MAE, R2, MaxAE",
}

CONDITIONS = {
    "S01_full_virtualnode_free_branch_degree_full_edge": {
        "aliases": ["A1", "B1", "C1"],
        "group_labels": [
            "Experiment A: edge-feature confirmation",
            "Experiment B: graph-construction confirmation",
            "Experiment C: node-feature confirmation",
        ],
        "condition_label": "full virtual-node baseline",
        "graph_label": "original bidirectional + virtual node",
        "node_features_label": "[is_free_end, is_branch_point, degree]",
        "edge_features_label": "[f, A/B]",
        "graph_builder": "virtual_node",
        "node_builder": "free_branch_degree",
        "edge_mode": "full",
        "optional": False,
    },
    "A2_virtualnode_free_branch_degree_edge_f_only": {
        "aliases": ["A2"],
        "group_labels": ["Experiment A: edge-feature confirmation"],
        "condition_label": "virtual node, edge f only",
        "graph_label": "original bidirectional + virtual node",
        "node_features_label": "[is_free_end, is_branch_point, degree]",
        "edge_features_label": "[f]",
        "graph_builder": "virtual_node",
        "node_builder": "free_branch_degree",
        "edge_mode": "f",
        "optional": False,
    },
    "A3_virtualnode_free_branch_degree_edge_type_only": {
        "aliases": ["A3"],
        "group_labels": ["Experiment A: edge-feature confirmation"],
        "condition_label": "virtual node, edge A/B only",
        "graph_label": "original bidirectional + virtual node",
        "node_features_label": "[is_free_end, is_branch_point, degree]",
        "edge_features_label": "[A/B]",
        "graph_builder": "virtual_node",
        "node_builder": "free_branch_degree",
        "edge_mode": "type",
        "optional": False,
    },
    "B2_bidirectional_free_branch_degree_full_edge": {
        "aliases": ["B2"],
        "group_labels": ["Experiment B: graph-construction confirmation"],
        "condition_label": "bidirectional without virtual node",
        "graph_label": "original bidirectional",
        "node_features_label": "[is_free_end, is_branch_point, degree]",
        "edge_features_label": "[f, A/B]",
        "graph_builder": "bidirectional",
        "node_builder": "free_branch_degree",
        "edge_mode": "full",
        "optional": False,
    },
    "B3_line_graph_free_branch_degree_full_edge": {
        "aliases": ["B3"],
        "group_labels": ["Experiment B: graph-construction confirmation"],
        "condition_label": "line graph",
        "graph_label": "line graph",
        "node_features_label": "line nodes from original edge [f, A/B]; line edges from original node [is_free_end, is_branch_point, degree]",
        "edge_features_label": "line edge_attr from original node semantics",
        "graph_builder": "line_graph",
        "node_builder": "free_branch_degree",
        "edge_mode": "full",
        "optional": False,
    },
    "B4_bipartite_free_branch_degree_full_edge": {
        "aliases": ["B4"],
        "group_labels": ["Experiment B: graph-construction confirmation"],
        "condition_label": "bipartite graph",
        "graph_label": "bipartite",
        "node_features_label": "junction=[is_free_end, is_branch_point, degree]; block=[f, A/B, block-indicator]",
        "edge_features_label": "bipartite connectivity edges are zero edge_attr",
        "graph_builder": "bipartite",
        "node_builder": "free_branch_degree",
        "edge_mode": "full",
        "optional": True,
    },
    "B5_self_loop_free_branch_degree_full_edge": {
        "aliases": ["B5"],
        "group_labels": ["Experiment B: graph-construction confirmation"],
        "condition_label": "self-loop graph",
        "graph_label": "original bidirectional + self-loop",
        "node_features_label": "[is_free_end, is_branch_point, degree]",
        "edge_features_label": "[f, A/B] with zero edge_attr on added self-loops",
        "graph_builder": "self_loop",
        "node_builder": "free_branch_degree",
        "edge_mode": "full",
        "optional": True,
    },
    "C2_virtualnode_learnable_node_full_edge": {
        "aliases": ["C2"],
        "group_labels": ["Experiment C: node-feature confirmation"],
        "condition_label": "virtual node with learnable node embeddings",
        "graph_label": "original bidirectional + virtual node",
        "node_features_label": "learnable node embedding; input data.x zeroed before encoder",
        "edge_features_label": "[f, A/B]",
        "graph_builder": "virtual_node",
        "node_builder": "learnable_node",
        "edge_mode": "full",
        "optional": True,
    },
    "C3_virtualnode_degree_avg_neighbor_full_edge": {
        "aliases": ["C3"],
        "group_labels": ["Experiment C: node-feature confirmation"],
        "condition_label": "virtual node with [degree, avg_neighbor_degree]",
        "graph_label": "original bidirectional + virtual node",
        "node_features_label": "[degree, avg_neighbor_degree]",
        "edge_features_label": "[f, A/B]",
        "graph_builder": "virtual_node",
        "node_builder": "degree_avg_neighbor_degree",
        "edge_mode": "full",
        "optional": True,
    },
}


def condition_config(condition_key: str) -> dict:
    if condition_key not in CONDITIONS:
        raise KeyError(f"Unknown condition: {condition_key}")
    cfg = dict(COMMON_CONFIG)
    cfg.update(CONDITIONS[condition_key])
    cfg["condition_key"] = condition_key
    return cfg


def build_note(cfg: dict) -> str:
    aliases = ", ".join(cfg["aliases"])
    groups = "; ".join(cfg["group_labels"])
    return (
        f"Supplementary confirmation condition {cfg['condition_key']} (aliases: {aliases}). "
        f"Groups: {groups}. "
        f"Graph: {cfg['graph_label']}. "
        f"Node features: {cfg['node_features_label']}. "
        f"Edge features: {cfg['edge_features_label']}."
    )


def build_wrapped_dataset(dataset, cfg: dict):
    node_builder = cfg["node_builder"]
    graph_builder = cfg["graph_builder"]
    edge_mode = cfg["edge_mode"]

    if node_builder == "free_branch_degree":
        prepared = base.FreeendBranchDegreeNodeFeatureWrapper(dataset)
    elif node_builder == "degree_avg_neighbor_degree":
        prepared = dataset
    elif node_builder == "learnable_node":
        prepared = dataset
    else:
        raise ValueError(f"Unknown node_builder: {node_builder}")

    if graph_builder == "virtual_node":
        node_mode = {
            "free_branch_degree": "original",
            "degree_avg_neighbor_degree": "degree_avg_neighbor_degree",
            "learnable_node": "zero",
        }[node_builder]
        return VirtualNodeWrapper(prepared, node_mode=node_mode, edge_mode=edge_mode, graph_mode="original")
    if graph_builder == "bidirectional":
        if node_builder != "free_branch_degree":
            raise ValueError("bidirectional graph_builder currently expects free_branch_degree node builder")
        return prepared
    if graph_builder == "line_graph":
        if node_builder != "free_branch_degree":
            raise ValueError("line_graph currently expects free_branch_degree node builder")
        return LineGraphWrapper(prepared, include_line_edge_attr=True)
    if graph_builder == "bipartite":
        if node_builder != "free_branch_degree":
            raise ValueError("bipartite currently expects free_branch_degree node builder")
        return BipartiteGraphWrapper(prepared, junction_mode="degree", block_mode="full")
    if graph_builder == "self_loop":
        if node_builder != "free_branch_degree":
            raise ValueError("self_loop currently expects free_branch_degree node builder")
        return SelfLoopGraphWrapper(prepared, node_mode="original")
    raise ValueError(f"Unknown graph_builder: {graph_builder}")


def load_wrapped_splits(cfg: dict):
    train_source = load_and_process_dataset(cfg["train_data_dir"], task_type="chiN")
    test_source = load_and_process_dataset(cfg["test_data_dir"], task_type="chiN")
    train_idx, val_idx = train_test_split(
        np.arange(len(train_source)),
        test_size=float(cfg["val_fraction"]),
        random_state=int(cfg["split_random_state"]),
    )
    train_subset = Subset(train_source, train_idx)
    val_subset = Subset(train_source, val_idx)

    train_dataset = build_wrapped_dataset(train_subset, cfg)
    val_dataset = build_wrapped_dataset(val_subset, cfg)
    test_dataset = build_wrapped_dataset(test_source, cfg)
    split_info = {
        "split_random_state": int(cfg["split_random_state"]),
        "train_source_size": int(len(train_source)),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_source)),
    }
    return train_dataset, val_dataset, test_dataset, split_info, train_idx, val_idx


def infer_feature_dims(dataset) -> tuple[int, int]:
    sample = dataset[0]
    in_feats = int(sample.x.size(1))
    edge_dim = int(sample.edge_attr.size(1)) if getattr(sample, "edge_attr", None) is not None else 0
    return in_feats, edge_dim


def build_model(cfg: dict, dataset, device: torch.device):
    if cfg["node_builder"] == "learnable_node":
        _, edge_dim = infer_feature_dims(dataset)
        encoder = LearnableNodeGINEEncoder(
            max_nodes=MAX_NODES_LEARNABLE,
            edge_dim=edge_dim,
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
        )
    else:
        in_feats, edge_dim = infer_feature_dims(dataset)
        encoder = GINEEncoder(
            in_feats=in_feats,
            edge_dim=edge_dim,
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
        )
    return SingleTaskRegressor(
        encoder,
        hidden_dim=int(cfg["hidden_dim"]),
        num_dec_layers=int(cfg["dec_layers"]),
    ).to(device)


def save_run_config(result_root: Path, cfg: dict, seed: int, run_id: int, device: torch.device, split_info: dict) -> None:
    payload = {
        "condition_key": cfg["condition_key"],
        "aliases": cfg["aliases"],
        "group_labels": cfg["group_labels"],
        "condition_label": cfg["condition_label"],
        "optional": bool(cfg["optional"]),
        "note": build_note(cfg),
        "paths": {
            "train_data_dir": cfg["train_data_dir"],
            "test_data_dir": cfg["test_data_dir"],
            "result_root": str(result_root),
        },
        "features": {
            "graph": cfg["graph_label"],
            "node_features": cfg["node_features_label"],
            "edge_features": cfg["edge_features_label"],
            "graph_builder": cfg["graph_builder"],
            "node_builder": cfg["node_builder"],
            "edge_mode": cfg["edge_mode"],
        },
        "hyperparameters": {
            "hidden_dim": int(cfg["hidden_dim"]),
            "learning_rate": float(cfg["learning_rate"]),
            "batch_size": int(cfg["batch_size"]),
            "num_layers": int(cfg["num_layers"]),
            "dec_layers": int(cfg["dec_layers"]),
            "epochs": int(cfg["epochs"]),
            "optimizer": cfg["optimizer"],
            "scheduler": cfg["scheduler"],
            "loss": cfg["loss"],
            "max_nodes_learnable": MAX_NODES_LEARNABLE if cfg["node_builder"] == "learnable_node" else None,
        },
        "runtime": {
            "seed": int(seed),
            "run_id": int(run_id),
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
    base.write_json(result_root / "run_config.json", payload)


def save_evaluation_outputs(model, test_loader, device, result_root: Path, cfg: dict, seed: int, run_id: int):
    model_path = result_root / "best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    y_pred, y_true = extract_predictions(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    base.save_predictions(y_true, y_pred, str(result_root))
    base.save_parity_plot_mpltex(
        y_true,
        y_pred,
        str(result_root / "parity_plot_chiN"),
        title=f"{cfg['condition_key']} seed {seed}",
        metrics=metrics,
    )
    result = {
        "condition_key": cfg["condition_key"],
        "aliases": ";".join(cfg["aliases"]),
        "condition_label": cfg["condition_label"],
        "group_labels": ";".join(cfg["group_labels"]),
        "optional": bool(cfg["optional"]),
        "run_id": int(run_id),
        "seed": int(seed),
        "MAPE": float(metrics["MAPE"]),
        "RMSE": float(metrics["RMSE"]),
        "MAE": float(metrics["MAE"]),
        "R2": float(metrics["R²"]),
        "MaxAE": float(metrics["MaxAE"]),
        "result_dir": str(result_root),
    }
    base.write_json(result_root / "metrics.json", result)
    report = (
        f"chiN supplementary confirmation: {cfg['condition_key']}\n"
        f"{'=' * 60}\n"
        f"Aliases: {', '.join(cfg['aliases'])}\n"
        f"Groups: {'; '.join(cfg['group_labels'])}\n"
        f"Run: {run_id}\n"
        f"Seed: {seed}\n"
        f"Graph: {cfg['graph_label']}\n"
        f"Node features: {cfg['node_features_label']}\n"
        f"Edge features: {cfg['edge_features_label']}\n"
        f"Hyperparameters:\n"
        f"  - hidden_dim: {cfg['hidden_dim']}\n"
        f"  - learning_rate: {cfg['learning_rate']}\n"
        f"  - batch_size: {cfg['batch_size']}\n"
        f"  - num_layers: {cfg['num_layers']}, dec_layers: {cfg['dec_layers']}\n"
        f"  - epochs: {cfg['epochs']}\n"
        f"Test Set Results:\n"
        f"  - MAPE : {metrics['MAPE']:.2f}%\n"
        f"  - RMSE : {metrics['RMSE']:.2f}\n"
        f"  - MAE  : {metrics['MAE']:.2f}\n"
        f"  - R2   : {metrics['R²']:.4f}\n"
        f"  - MaxAE: {metrics['MaxAE']:.2f}\n"
        f"{'=' * 60}\n"
        f"{build_note(cfg)}\n"
    )
    with open(result_root / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    return result


def train_one(condition_key: str, batch_dir: str, seed: int, run_id: int):
    cfg = condition_config(condition_key)
    result_root = Path(batch_dir) / "runs" / condition_key / f"{condition_key}_run{run_id:02d}_seed{seed}"
    base.set_seed(seed)
    result_root.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, test_dataset, split_info, train_idx, val_idx = load_wrapped_splits(cfg)
    np.savez(
        result_root / "split_indices.npz",
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=int(cfg["batch_size"]), shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=int(cfg["batch_size"]))
    test_loader = DataLoader(test_dataset, batch_size=int(cfg["batch_size"]))

    model = build_model(cfg, train_dataset, device)
    base.save_model_artifacts(model, str(result_root))
    save_run_config(result_root, cfg, seed, run_id, device, split_info)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(cfg["scheduler"]["mode"]),
        factor=float(cfg["scheduler"]["factor"]),
        patience=int(cfg["scheduler"]["patience"]),
    )
    criterion = nn.MSELoss()
    model_path = result_root / "best_model.pth"
    trainer = Trainer(model, optimizer, criterion, device, scheduler=scheduler, save_path=str(model_path))
    train_hist, val_hist = trainer.fit(train_loader, val_loader, epochs=int(cfg["epochs"]))
    base.save_loss_history(train_hist, val_hist, str(result_root))
    base.save_loss_curve_mpltex(train_hist, val_hist, str(result_root / "loss_curve"))

    return save_evaluation_outputs(model, test_loader, device, result_root, cfg, seed, run_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-key", required=True)
    parser.add_argument("--batch-dir", default=str(DEFAULT_BATCH_ROOT))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    train_one(args.condition_key, args.batch_dir, args.seed, args.run_id)


if __name__ == "__main__":
    main()
