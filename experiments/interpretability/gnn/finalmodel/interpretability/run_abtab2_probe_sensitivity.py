#!/usr/bin/env python3
"""PCA/Ridge sensitivity analysis for the final-C2 ABTAB2 layer probe.

Every reported score is an out-of-fold prediction from a strict five-fold
leave-one-rho-out split.  Standardization and PCA are refit inside each outer
training fold.  The sensitivity grid is descriptive: no setting is selected
from the held-out scores.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results"
DEFAULT_FIGURES = HERE.parent / "figure" / "c2_model" / "interpretability_ablation"
SKILL_SCRIPTS = Path("/home/zxy/.codex/skills/mpltex-skill/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from paper_style import COLORS, configure_paper_style, save_dual, style_axis  # noqa: E402


configure_paper_style()

PCA_COMPONENTS = (2, 3, 4, 6, 8)
RIDGE_ALPHAS = (10.0, 100.0, 1000.0)
CONCEPTS = ("S_AA", "S_BB", "S_AB", "q_star", "chiN_star")
STAGES = ("Input", "GINE-1", "GINE-2", "GINE-3", "GINE-4", "GINE-5")


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def run_sensitivity(
    results_dir: Path,
    pca_components: tuple[int, ...],
    ridge_alphas: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(results_dir / "abtab2_layerwise_embeddings.csv")
    expected_rho = {0.5, 1.0, 2.0, 5.0, 10.0}
    if set(frame.rho.unique()) != expected_rho:
        raise ValueError("Expected the five prescribed rho groups.")
    if not (frame.groupby("rho").f_A.nunique() == 9).all():
        raise ValueError("Each rho group must contain all nine f_A values.")

    groups = frame.rho.to_numpy()
    folds = list(GroupKFold(n_splits=5).split(frame, groups=groups))
    aggregate_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for layer in range(6):
        x = np.stack(frame[f"embedding_L{layer}"].map(json.loads))
        for n_components in pca_components:
            transformed = []
            for train_idx, test_idx in folds:
                scaler = StandardScaler()
                train_scaled = scaler.fit_transform(x[train_idx])
                test_scaled = scaler.transform(x[test_idx])
                pca = PCA(n_components=n_components, whiten=True, svd_solver="full")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    train_reduced = pca.fit_transform(train_scaled)
                    test_reduced = pca.transform(test_scaled)
                transformed.append((train_idx, test_idx, train_reduced, test_reduced))

            for alpha in ridge_alphas:
                for concept in CONCEPTS:
                    y = frame[concept].to_numpy(dtype=float)
                    prediction = np.empty_like(y)
                    local_fold_scores = []
                    for fold, (train_idx, test_idx, z_train, z_test) in enumerate(
                        transformed, start=1
                    ):
                        model = Ridge(alpha=alpha)
                        model.fit(z_train, y[train_idx])
                        prediction[test_idx] = model.predict(z_test)
                        held_out_rho = float(groups[test_idx][0])
                        fold_r2 = float(r2_score(y[test_idx], prediction[test_idx]))
                        fold_rmse = float(
                            mean_squared_error(y[test_idx], prediction[test_idx]) ** 0.5
                        )
                        local_fold_scores.append(fold_r2)
                        fold_rows.append({
                            "pca_components": n_components,
                            "ridge_alpha": alpha,
                            "layer": layer,
                            "concept": concept,
                            "fold": fold,
                            "held_out_rho": held_out_rho,
                            "R2_fold": fold_r2,
                            "RMSE_fold": fold_rmse,
                        })
                        for idx in test_idx:
                            prediction_rows.append({
                                "pca_components": n_components,
                                "ridge_alpha": alpha,
                                "layer": layer,
                                "concept": concept,
                                "fold": fold,
                                "held_out_rho": held_out_rho,
                                "f_A": float(frame.iloc[idx].f_A),
                                "y_true": float(y[idx]),
                                "y_pred": float(prediction[idx]),
                            })
                    aggregate_rows.append({
                        "pca_components": n_components,
                        "ridge_alpha": alpha,
                        "layer": layer,
                        "concept": concept,
                        "n": len(y),
                        "R2_leave_one_rho_out": float(r2_score(y, prediction)),
                        "RMSE_leave_one_rho_out": float(
                            mean_squared_error(y, prediction) ** 0.5
                        ),
                        "median_fold_R2": float(np.median(local_fold_scores)),
                        "minimum_fold_R2": float(np.min(local_fold_scores)),
                    })

    aggregate = pd.DataFrame(aggregate_rows)
    fold_scores = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    aggregate.to_csv(results_dir / "abtab2_layerwise_probe_sensitivity.csv", index=False)
    fold_scores.to_csv(results_dir / "abtab2_layerwise_probe_sensitivity_folds.csv", index=False)
    predictions.to_csv(
        results_dir / "abtab2_layerwise_probe_sensitivity_predictions.csv", index=False
    )
    return aggregate, fold_scores, predictions


def summarize(aggregate: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    layer_summary = (
        aggregate.groupby(["pca_components", "ridge_alpha", "layer"], as_index=False)
        .agg(
            mean_R2=("R2_leave_one_rho_out", "mean"),
            median_R2=("R2_leave_one_rho_out", "median"),
            minimum_R2=("R2_leave_one_rho_out", "min"),
            maximum_R2=("R2_leave_one_rho_out", "max"),
            positive_concepts=("R2_leave_one_rho_out", lambda value: int((value > 0).sum())),
        )
    )
    layer_summary.to_csv(
        results_dir / "abtab2_layerwise_probe_sensitivity_layer_summary.csv", index=False
    )

    setting_rows = []
    for (components, alpha), group in layer_summary.groupby(
        ["pca_components", "ridge_alpha"], sort=True
    ):
        best_mean = group.loc[group.mean_R2.idxmax()]
        best_median = group.loc[group.median_R2.idxmax()]
        setting_rows.append({
            "pca_components": int(components),
            "ridge_alpha": float(alpha),
            "best_layer_by_mean_R2": int(best_mean.layer),
            "best_mean_R2": float(best_mean.mean_R2),
            "best_layer_by_median_R2": int(best_median.layer),
            "best_median_R2": float(best_median.median_R2),
            "GINE2_3_best_by_mean": int(best_mean.layer in (2, 3)),
            "Input_or_GINE1_best_by_mean": int(best_mean.layer in (0, 1)),
        })
    settings = pd.DataFrame(setting_rows)
    settings.to_csv(
        results_dir / "abtab2_layerwise_probe_sensitivity_setting_summary.csv", index=False
    )
    payload = {
        "n_settings": len(settings),
        "pca_components": sorted(aggregate.pca_components.unique().tolist()),
        "ridge_alphas": sorted(aggregate.ridge_alpha.unique().tolist()),
        "best_layer_by_mean_counts": {
            STAGES[int(layer)]: int(count)
            for layer, count in settings.best_layer_by_mean_R2.value_counts().sort_index().items()
        },
        "GINE2_or_GINE3_best_by_mean_count": int(settings.GINE2_3_best_by_mean.sum()),
        "Input_or_GINE1_best_by_mean_count": int(settings.Input_or_GINE1_best_by_mean.sum()),
        "interpretation": (
            "Sensitivity analysis only; outer leave-one-rho-out scores were not used "
            "to select the canonical PCA(2)+Ridge(10) probe."
        ),
    }
    (results_dir / "abtab2_layerwise_probe_sensitivity_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return layer_summary


def plot_sensitivity(
    layer_summary: pd.DataFrame,
    figure_dir: Path,
    pca_components: tuple[int, ...],
    ridge_alphas: tuple[float, ...],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "tapd_probe_sensitivity", [COLORS["blue"], "#FFFFFF", COLORS["red"]]
    )
    fig, axes = plt.subplots(1, len(ridge_alphas), figsize=(21.0, 8.5), sharey=True)
    if len(ridge_alphas) == 1:
        axes = [axes]
    image = None
    for panel, (ax, alpha) in enumerate(zip(axes, ridge_alphas)):
        subset = layer_summary[np.isclose(layer_summary.ridge_alpha, alpha)]
        grid = (
            subset.pivot(index="pca_components", columns="layer", values="mean_R2")
            .reindex(index=pca_components, columns=range(6))
        )
        values = grid.to_numpy()
        image = ax.imshow(values, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(6), STAGES, rotation=38, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(pca_components)), [str(value) for value in pca_components])
        ax.set_xlabel("Representation stage", fontsize=24)
        if panel == 0:
            ax.set_ylabel("PCA components", fontsize=26)
        ax.text(
            0.5,
            1.035,
            rf"Ridge $\alpha={alpha:g}$",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=27,
        )
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                label = f"{value:.2f}" if value >= -9.99 else "< -10"
                color = "#FFFFFF" if abs(np.clip(value, -1, 1)) >= 0.58 else COLORS["black"]
                ax.text(col, row, label, ha="center", va="center", fontsize=16, color=color)
        style_axis(ax)
        ax.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False)
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=21)

    if image is None:
        raise RuntimeError("No sensitivity panels were drawn.")
    fig.subplots_adjust(left=0.075, right=0.885, bottom=0.24, top=0.89, wspace=0.18)
    cbar_axis = fig.add_axes([0.915, 0.24, 0.014, 0.65])
    cbar = fig.colorbar(image, cax=cbar_axis)
    cbar.set_label(r"Mean concept-probe $R^2$", fontsize=25, labelpad=12)
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    style_axis(cbar.ax)
    cbar.ax.tick_params(labelsize=20, direction="in", length=8, width=1.8)
    cbar.outline.set_edgecolor(COLORS["black"])
    cbar.outline.set_linewidth(2.0)
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_dual(fig, figure_dir / "abtab2_layerwise_probe_sensitivity", tight=False)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--pca-components", default="2,3,4,6,8")
    parser.add_argument("--ridge-alphas", default="10,100,1000")
    args = parser.parse_args()
    components = parse_ints(args.pca_components)
    alphas = parse_floats(args.ridge_alphas)
    aggregate, _, _ = run_sensitivity(args.results_dir, components, alphas)
    layer_summary = summarize(aggregate, args.results_dir)
    plot_sensitivity(layer_summary, args.figure_dir, components, alphas)
    print(
        f"Completed {len(components) * len(alphas)} sensitivity settings; "
        f"results={args.results_dir}; figures={args.figure_dir}"
    )


if __name__ == "__main__":
    main()
