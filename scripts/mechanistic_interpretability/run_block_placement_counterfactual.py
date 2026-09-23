#!/usr/bin/env python3
"""Reproduce the five-block block-placement counterfactual experiment.

For each of 1,170 block-attribute assignments this script enumerates all 132
generator-valid growth histories, averages the six layer embeddings, computes
all 16,065 within-condition assignment pairs, and records every prediction.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
from experiment_support import (
    blocks_for,
    build_model,
    legal_growth_orders,
    make_five_block_graph,
    path_concepts,
    unique_assignments,
)

DEFAULT_RUN = CODE_ROOT / "checkpoints/gnn/seed202601"
DEFAULT_OUTPUT = CODE_ROOT / "data/mechanistic_interpretability/legal_histories"


def infer_layers(model, graphs, device: torch.device, batch_size: int) -> list[np.ndarray]:
    output: list[list[np.ndarray]] = [[] for _ in range(6)]
    model.eval()
    with torch.no_grad():
        for data in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            data = data.to(device)
            encoder = model.encoder
            counts = (data.ptr[1:] - data.ptr[:-1]).tolist()
            x = torch.cat([encoder.node_embedding[:count] for count in counts])
            # The layer-0 value is the mean of the same positional table for
            # every 33-node graph. Computing each mean directly avoids tiny
            # CUDA scatter-order noise that would create a spurious L0 rank.
            output[0].append(
                torch.stack([encoder.node_embedding[:count].mean(dim=0) for count in counts])
                .cpu().numpy()
            )
            for layer, convolution in enumerate(encoder.gine.convs):
                x = convolution(
                    x,
                    data.edge_index,
                    edge_attr=encoder.gine.edge_lin[layer](data.edge_attr),
                )
                if layer != len(encoder.gine.convs) - 1:
                    x = encoder.gine.relu(x)
                output[layer + 1].append(global_mean_pool(x, data.batch).cpu().numpy())
    return [np.concatenate(items) for items in output]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-conditions", type=int,
                        help="Development smoke test only; omit for the 45-condition paper run")
    parser.add_argument("--max-assignments", type=int,
                        help="Development smoke test only; omit for all assignments")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    model = build_model(config, device)
    model.load_state_dict(
        torch.load(args.run_dir / "best_model.pth", map_location=device, weights_only=True),
        strict=True,
    )
    model.eval()

    reference_blocks = blocks_for(0.5, 1.0)
    histories = legal_growth_orders(reference_blocks)
    if len(histories) != 132:
        raise AssertionError(f"Expected 132 legal histories, got {len(histories)}")
    history_rows = [
        {"history_id": index, "node_order": json.dumps(order, separators=(",", ":"))}
        for index, order in enumerate(histories)
    ]
    write_csv(args.output_dir / "legal_growth_orders.csv", history_rows)

    condition_grid = [
        (float(f_a), float(rho))
        for f_a in np.round(np.linspace(0.1, 0.9, 9), 2)
        for rho in (0.5, 1.0, 2.0, 5.0, 10.0)
    ]
    if args.max_conditions is not None:
        condition_grid = condition_grid[: args.max_conditions]

    assignment_rows: list[dict] = []
    pair_rows: list[dict] = []
    stability_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for group, (f_a, rho) in enumerate(condition_grid):
        assignments = unique_assignments(f_a, rho)
        if args.max_assignments is not None:
            assignments = assignments[: args.max_assignments]
        means: list[list[np.ndarray]] = []
        concepts: list[np.ndarray] = []
        for assignment_index, assignment in enumerate(assignments):
            graphs = [
                make_five_block_graph(f_a, rho, assignment, order)
                for order in histories
            ]
            layers = infer_layers(model, graphs, device, args.batch_size)
            with torch.no_grad():
                predictions = (
                    model.decoder(torch.as_tensor(layers[5], device=device))
                    .detach().cpu().numpy().reshape(-1)
                )
            layer_means = [values.mean(axis=0) for values in layers]
            concept = path_concepts(f_a, rho, assignment)
            means.append(layer_means)
            concepts.append(concept)
            relative = np.abs(np.power(10.0, predictions - predictions.mean()) - 1.0) * 100.0
            stability_rows.append({
                "group": group,
                "f_A": f_a,
                "rho": rho,
                "assignment": assignment_index,
                "n_legal_histories": len(histories),
                "prediction_mean_log10": float(predictions.mean()),
                "prediction_sd_log10": float(predictions.std()),
                "prediction_range_log10": float(predictions.max() - predictions.min()),
                "mean_abs_relative_to_history_mean_percent": float(relative.mean()),
                "max_abs_relative_to_history_mean_percent": float(relative.max()),
            })
            assignment_rows.append({
                "group": group,
                "f_A": f_a,
                "rho": rho,
                "assignment": assignment_index,
                "assignment_species_and_lengths": json.dumps(assignment, separators=(",", ":")),
                "n_legal_histories": len(histories),
                "S_AA_separated": float(concept[0]),
                "S_BB_separated": float(concept[1]),
                "S_AB_separated": float(concept[2]),
                "q_star": float(concept[3]),
                **{
                    f"embedding_L{layer}": json.dumps(layer_means[layer].tolist(), separators=(",", ":"))
                    for layer in range(6)
                },
            })
            for history_id, prediction in enumerate(predictions):
                prediction_rows.append({
                    "group": group,
                    "f_A": f_a,
                    "rho": rho,
                    "assignment": assignment_index,
                    "history_id": history_id,
                    "prediction_log10_chiN": float(prediction),
                    "prediction_chiN": float(10.0 ** prediction),
                })
        for left, right in itertools.combinations(range(len(assignments)), 2):
            pair_rows.append({
                "group": group,
                "i": left,
                "j": right,
                "concept": float(np.linalg.norm(concepts[left] - concepts[right])),
                **{
                    f"embedding_L{layer}": float(np.linalg.norm(means[left][layer] - means[right][layer]))
                    for layer in range(6)
                },
            })
        print(f"completed {group + 1}/{len(condition_grid)}", flush=True)

    assignments_frame = pd.DataFrame(assignment_rows)
    pairs_frame = pd.DataFrame(pair_rows)
    stability_frame = pd.DataFrame(stability_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    assignments_frame.to_csv(args.output_dir / "matched_path_history_mean_embeddings.csv", index=False)
    pairs_frame.to_csv(args.output_dir / "matched_path_legal_history_pairs.csv", index=False)
    stability_frame.to_csv(args.output_dir / "matched_path_legal_history_stability.csv", index=False)
    predictions_frame.to_csv(args.output_dir / "per_history_predictions.csv", index=False)
    summary = [
        {
            "layer": layer,
            "n_pairs": len(pairs_frame),
            "spearman_mean_embedding_vs_RPA_path": float(
                spearmanr(pairs_frame[f"embedding_L{layer}"], pairs_frame["concept"]).statistic
            ),
        }
        for layer in range(6)
    ]
    payload = {
        "checkpoint": str(args.run_dir.resolve()),
        "history_definition": "all node orders satisfying the generator's sequential tree-growth rule",
        "n_conditions": len(condition_grid),
        "n_assignments": len(assignments_frame),
        "n_histories_per_assignment": len(histories),
        "n_prediction_rows": len(predictions_frame),
        "summary": summary,
        "prediction_stability": {
            "mean_abs_relative_percent": float(stability_frame.mean_abs_relative_to_history_mean_percent.mean()),
            "p95_max_abs_relative_percent": float(stability_frame.max_abs_relative_to_history_mean_percent.quantile(0.95)),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
