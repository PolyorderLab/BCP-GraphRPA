#!/usr/bin/env python3
"""Fit low-capacity ABTAB2 layer probes with leave-one-rho-out validation.

This script consumes the deterministic frozen-C2 layer embeddings.  Each fold
holds out all nine ``f_A`` samples at one value of ``rho``; it never splits an
individual rho group between training and test data.  Besides the aggregate
score, it records every out-of-fold prediction, making numerical failures
visible rather than hidden in a clipped layerwise figure.  Scaling and PCA are
fit independently inside each training fold, so neither transformation sees
the held-out rho group.  A fixed two-component whitened PCA and fixed Ridge
penalty avoid attempting hyperparameter selection from only four inner groups.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parents[1] / "data/mechanistic_interpretability/rpa"
CONCEPTS = ("S_AA", "S_BB", "S_AB", "q_star", "chiN_star")


def fit_probe(
    results_dir: Path = DEFAULT_RESULTS,
    pca_components: int = 2,
    ridge_alpha: float = 10.0,
) -> list[dict[str, object]]:
    """Fit the canonical probe from an existing layer-embedding table."""
    frame = pd.read_csv(results_dir / "abtab2_layerwise_embeddings.csv")
    expected_rho = {0.5, 1.0, 2.0, 5.0, 10.0}
    if set(frame["rho"].unique()) != expected_rho:
        raise ValueError("Expected exactly the five prescribed rho groups.")
    if not (frame.groupby("rho")["f_A"].nunique() == 9).all():
        raise ValueError("Each rho group must contain the complete nine-point f_A grid.")

    groups = frame["rho"].to_numpy()
    split = list(GroupKFold(n_splits=5).split(frame, groups=groups))
    prediction_rows, summary_rows, selection_rows = [], [], []
    for layer in range(6):
        x = np.stack(frame[f"embedding_L{layer}"].map(json.loads))
        for concept in CONCEPTS:
            y = frame[concept].to_numpy(dtype=float)
            prediction = np.empty_like(y)
            for fold, (train_idx, test_idx) in enumerate(split, start=1):
                held_out_rho = float(groups[test_idx][0])
                if not np.all(groups[test_idx] == held_out_rho):
                    raise AssertionError("A validation fold mixed rho groups.")
                model = Pipeline([
                    ("scale", StandardScaler()),
                    ("pca", PCA(
                        n_components=pca_components,
                        whiten=True,
                        svd_solver="full",
                    )),
                    ("ridge", Ridge(alpha=ridge_alpha)),
                ])
                model.fit(x[train_idx], y[train_idx])
                prediction[test_idx] = model.predict(x[test_idx])
                selection_rows.append({
                    "layer": layer, "concept": concept, "fold": fold,
                    "held_out_rho": held_out_rho,
                    "pca_components": pca_components,
                    "ridge_alpha": ridge_alpha,
                })
                for idx in test_idx:
                    prediction_rows.append({
                        "layer": layer, "concept": concept, "fold": fold,
                        "held_out_rho": held_out_rho, "f_A": float(frame.iloc[idx]["f_A"]),
                        "y_true": float(y[idx]), "y_pred": float(prediction[idx]),
                    })
            summary_rows.append({
                "layer": layer, "concept": concept, "n_total": len(y),
                "n_train_per_fold": 36, "n_test_per_fold": 9,
                "validation": (
                    f"leave-one-rho-out PCA({pca_components}, whitened)"
                    f"+Ridge({ridge_alpha:g})"
                ),
                "R2_grouped_by_rho": float(r2_score(y, prediction)),
                "RMSE_grouped_by_rho": float(mean_squared_error(y, prediction) ** 0.5),
            })

    pd.DataFrame(prediction_rows).to_csv(
        results_dir / "abtab2_layerwise_pca_ridge_predictions.csv", index=False
    )
    pd.DataFrame(selection_rows).to_csv(
        results_dir / "abtab2_layerwise_pca_ridge_folds.csv", index=False
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results_dir / "abtab2_layerwise_pca_ridge_probe.csv", index=False)
    # This is the canonical input consumed by the layerwise figure plotter.
    summary[["layer", "concept", "n_total", "R2_grouped_by_rho"]].rename(
        columns={"n_total": "n"}
    ).to_csv(results_dir / "abtab2_layerwise_probe.csv", index=False)
    (results_dir / "abtab2_layerwise_pca_ridge_protocol.json").write_text(
        json.dumps(
            {
                "outer_validation": "GroupKFold(n_splits=5, groups=rho)",
                "rho_groups": sorted(expected_rho),
                "samples_per_test_fold": 9,
                "samples_per_train_fold": 36,
                "pipeline_fit_inside_each_outer_train_fold": [
                    "StandardScaler",
                    f"PCA(n_components={pca_components}, whiten=True)",
                    f"Ridge(alpha={ridge_alpha})",
                ],
                "hyperparameters_fixed_before_outer_evaluation": True,
                "embedding_dimension_before_pca": int(
                    len(json.loads(frame.iloc[0]["embedding_L1"]))
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote leave-one-rho-out PCA+Ridge probe to {results_dir}")
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--pca-components", type=int, default=2)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    args = parser.parse_args()
    fit_probe(args.results_dir, args.pca_components, args.ridge_alpha)


if __name__ == "__main__":
    main()
