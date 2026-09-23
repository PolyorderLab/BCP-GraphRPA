#!/usr/bin/env python3
"""Reproduce the Seed-43 legal-growth-history sensitivity experiment.

The output contains one row for every selected graph, frozen checkpoint, and
history (the stored order plus three alternative generator-valid orders).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
from experiment_support import VirtualNodeWrapper, build_model, load_seed43_dataset


DEFAULT_TEST_DIR = CODE_ROOT / "data/datasets/seed43_2k"
DEFAULT_CHECKPOINT_ROOT = CODE_ROOT / "checkpoints/gnn"
DEFAULT_OUTPUT = CODE_ROOT / "data/mechanistic_interpretability/seed43"
SEEDS = tuple(range(202601, 202606))


def valid_orders(data) -> list[tuple[int, ...]]:
    """Enumerate all sequential tree-growth orders of the physical graph."""
    physical = sorted(set(data.edge_index.reshape(-1).tolist()))
    if physical != list(range(len(physical))):
        raise ValueError("Expected physical nodes in a contiguous zero-based prefix")
    adjacency = {node: set() for node in physical}
    for left, right in data.edge_index.t().tolist():
        adjacency[left].add(right)
        adjacency[right].add(left)
    orders: list[tuple[int, ...]] = []

    def extend(order: list[int], seen: set[int]) -> None:
        if len(order) == len(physical):
            orders.append(tuple(order))
            return
        frontier = sorted({v for u in seen for v in adjacency[u]} - seen)
        for node in frontier:
            extend(order + [node], seen | {node})

    for root in physical:
        extend([root], {root})
    return orders


def relabel_to_history(data, order: tuple[int, ...]):
    """Reassign physical-node IDs while retaining padding and graph semantics."""
    out = data.clone()
    n_physical = len(order)
    slots = data.x.size(0)
    old_to_new = torch.arange(slots, dtype=torch.long)
    old_to_new[torch.tensor(order)] = torch.arange(n_physical, dtype=torch.long)
    out.x = data.x.clone()
    out.x[:n_physical] = data.x[torch.tensor(order)]
    out.edge_index = old_to_new[data.edge_index].clone()
    return out


def predict(model, dataset, device: torch.device, batch_size: int) -> np.ndarray:
    values = []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            values.append(model(batch.to(device)).detach().cpu().numpy().reshape(-1))
    return np.concatenate(values)


def summarize(base: np.ndarray, alternative: np.ndarray) -> dict[str, float]:
    delta = alternative - base
    relative = np.abs(np.power(10.0, delta) - 1.0) * 100.0
    return {
        "mean_abs_relative_change_percent": float(relative.mean()),
        "median_abs_relative_change_percent": float(np.median(relative)),
        "p95_abs_relative_change_percent": float(np.quantile(relative, 0.95)),
        "max_abs_relative_change_percent": float(relative.max()),
        "fraction_gt_1pct": float(np.mean(relative > 1.0)),
        "fraction_gt_5pct": float(np.mean(relative > 5.0)),
        "mean_abs_delta_log10": float(np.abs(delta).mean()),
    }


def checkpoint_records(root: Path) -> list[tuple[int, dict, Path]]:
    records = []
    for seed in SEEDS:
        directory = root / f"seed{seed}"
        metrics = directory / "metrics.json"
        checkpoint = directory / "best_model.pth"
        if not metrics.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"Incomplete checkpoint directory: {directory}")
        records.append((seed, json.loads(metrics.read_text(encoding="utf-8")), checkpoint))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--histories", type=int, default=4,
                        help="Total histories per graph, including the stored order")
    parser.add_argument("--rng-seed", type=int, default=20260904)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.histories < 2:
        raise ValueError("--histories must include the original and at least one alternative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "seed43_parsed_graphs.pt"
    if cache.is_file():
        raw_all = torch.load(cache, map_location="cpu", weights_only=False)
    else:
        raw_all = load_seed43_dataset(args.test_dir)
        torch.save(raw_all, cache)
    source_files = [path.stem for path in sorted(args.test_dir.glob("*.mat"))]
    if len(source_files) != len(raw_all):
        raise RuntimeError("Parsed graph count differs from sorted MAT-file count")

    selected = np.linspace(0, len(raw_all) - 1, min(args.samples, len(raw_all)), dtype=int)
    raw = [raw_all[int(index)] for index in selected]
    rng = np.random.default_rng(args.rng_seed)
    all_orders: list[list[tuple[int, ...]]] = []
    history_graphs: list[list] = [[] for _ in range(args.histories)]
    manifest_rows = []
    history_counts = []
    for sample_position, (source_index, graph) in enumerate(zip(selected, raw, strict=True)):
        orders = valid_orders(graph)
        original = tuple(range(len(orders[0])))
        if original not in orders:
            raise AssertionError("Stored node order is not a valid generator history")
        candidates = [order for order in orders if order != original]
        take = args.histories - 1
        if len(candidates) < take:
            raise RuntimeError(f"Graph {source_index} has only {len(candidates)} alternative histories")
        chosen = rng.choice(len(candidates), size=take, replace=False)
        selected_orders = [original] + [candidates[int(index)] for index in chosen]
        all_orders.append(selected_orders)
        history_counts.append(len(orders))
        for history_id, order in enumerate(selected_orders):
            history_graphs[history_id].append(graph if history_id == 0 else relabel_to_history(graph, order))
            manifest_rows.append({
                "sample_position": sample_position,
                "source_index": int(source_index),
                "source_file": source_files[int(source_index)],
                "history_id": history_id,
                "is_original": history_id == 0,
                "node_order_new_to_old": json.dumps(order, separators=(",", ":")),
                "n_valid_histories": len(orders),
            })

    checkpoints = checkpoint_records(args.checkpoint_root)
    wrapped = [VirtualNodeWrapper(graphs) for graphs in history_graphs]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    summary_rows = []
    prediction_rows = []
    for seed, config, checkpoint in checkpoints:
        model = build_model(config, device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        predictions = np.stack([predict(model, dataset, device, args.batch_size) for dataset in wrapped])
        base = predictions[0]
        alternatives = predictions[1:]
        row = {"seed": seed, **summarize(np.tile(base, args.histories - 1), alternatives.reshape(-1))}
        summary_rows.append(row)
        for sample_position, source_index in enumerate(selected):
            truth_log10 = float(raw[sample_position].y.reshape(-1)[0])
            for history_id in range(args.histories):
                pred_log10 = float(predictions[history_id, sample_position])
                delta_log10 = pred_log10 - float(base[sample_position])
                prediction_rows.append({
                    "checkpoint_seed": seed,
                    "sample_position": sample_position,
                    "source_index": int(source_index),
                    "source_file": source_files[int(source_index)],
                    "history_id": history_id,
                    "is_original": history_id == 0,
                    "node_order_new_to_old": json.dumps(all_orders[sample_position][history_id], separators=(",", ":")),
                    "n_valid_histories": history_counts[sample_position],
                    "true_log10_chiN": truth_log10,
                    "true_chiN": 10.0 ** truth_log10,
                    "prediction_log10_chiN": pred_log10,
                    "prediction_chiN": 10.0 ** pred_log10,
                    "baseline_prediction_log10_chiN": float(base[sample_position]),
                    "delta_log10": delta_log10,
                    "abs_relative_change_percent": abs(10.0 ** delta_log10 - 1.0) * 100.0,
                })
        print(json.dumps(row), flush=True)

    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.output_dir / "history_manifest.csv", manifest_rows)
    write_csv(args.output_dir / "per_sample_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "per_checkpoint_summary.csv", summary_rows)
    aggregate = {
        key: float(np.mean([row[key] for row in summary_rows]))
        for key in summary_rows[0]
        if key != "seed"
    }
    payload = {
        "protocol": "500 evenly spaced Seed-43 graphs; stored order plus three sampled alternative generator-valid growth histories; five frozen C2 checkpoints.",
        "test_source": str(args.test_dir.resolve()),
        "sample_indices": selected.tolist(),
        "n_graphs": len(raw),
        "histories_per_graph": args.histories,
        "n_checkpoints": len(checkpoints),
        "n_prediction_rows": len(prediction_rows),
        "valid_history_order_count": sorted(set(history_counts)),
        "rng_seed": args.rng_seed,
        "device": str(device),
        "per_checkpoint": summary_rows,
        "mean_over_five_checkpoints": aggregate,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
