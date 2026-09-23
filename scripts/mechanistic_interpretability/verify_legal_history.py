#!/usr/bin/env python3
"""Validate the two archived legal-growth-history experiments."""
from __future__ import annotations

import csv
import json
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]


def data_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def main() -> None:
    legal = CODE_ROOT / "data/mechanistic_interpretability/legal_histories"
    seed43 = CODE_ROOT / "data/mechanistic_interpretability/seed43"
    expected = {
        legal / "legal_growth_orders.csv": 132,
        legal / "matched_path_history_mean_embeddings.csv": 1170,
        legal / "matched_path_legal_history_pairs.csv": 16065,
        legal / "matched_path_legal_history_stability.csv": 1170,
        legal / "per_history_predictions.csv": 1170 * 132,
        seed43 / "history_manifest.csv": 500 * 4,
        seed43 / "per_sample_predictions.csv": 500 * 4 * 5,
        seed43 / "per_checkpoint_summary.csv": 5,
    }
    for path, count in expected.items():
        observed = data_rows(path)
        if observed != count:
            raise AssertionError(f"{path.name}: expected {count} rows, found {observed}")

    inputs = CODE_ROOT / "data/datasets/seed43_2k"
    mat_count = len(list(inputs.glob("*.mat")))
    yaml_count = len(list(inputs.glob("*.yml")))
    if (mat_count, yaml_count) != (2000, 2000):
        raise AssertionError(f"Seed-43 inputs: expected 2000 MAT/YML pairs, found {mat_count}/{yaml_count}")
    checkpoints = list((CODE_ROOT / "checkpoints/gnn").glob("seed20260[1-5]/best_model.pth"))
    configs = list((CODE_ROOT / "checkpoints/gnn").glob("seed20260[1-5]/metrics.json"))
    if len(checkpoints) != 5 or len(configs) != 5:
        raise AssertionError("Expected five checkpoint/config pairs")

    legal_summary = json.loads((legal / "summary.json").read_text(encoding="utf-8"))
    seed_summary = json.loads((seed43 / "summary.json").read_text(encoding="utf-8"))
    assert legal_summary["n_conditions"] == 45
    assert legal_summary["n_assignments"] == 1170
    assert legal_summary["n_histories_per_assignment"] == 132
    assert legal_summary["n_prediction_rows"] == 154440
    assert legal_summary["summary"][-1]["n_pairs"] == 16065
    assert abs(legal_summary["summary"][-1]["spearman_mean_embedding_vs_RPA_path"] - 0.77513) < 1e-4
    assert seed_summary["n_graphs"] == 500
    assert seed_summary["histories_per_graph"] == 4
    assert seed_summary["n_checkpoints"] == 5
    assert seed_summary["n_prediction_rows"] == 10000
    print("Package validation passed: block-placement and Seed-43 artifacts are complete.")


if __name__ == "__main__":
    main()
