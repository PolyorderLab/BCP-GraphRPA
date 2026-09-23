#!/usr/bin/env python3
"""Draw presentation panels (a)--(d) from the final-C2 ABTAB2 rerun."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path("/home/zxy/project/TAPD")
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "mainmodel/interpretability/abtab2_rpa_gnn/plot_abtab2_three_panel_mpltex.py"

spec = importlib.util.spec_from_file_location("abtab2_plot_source", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load plot source: {SOURCE}")
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)
plots.RESULTS = HERE / "results"

# Keep final-C2 interpretability panels together in the organized figure tree.
figure_dir = HERE.parent / "figure" / "c2_model" / "interpretability_ablation"
figure_dir.mkdir(parents=True, exist_ok=True)
original_save_dual = plots.save_dual
plots.save_dual = lambda fig, stem, **kwargs: original_save_dual(fig, figure_dir / Path(stem).name, **kwargs)


def plot_layerwise_pca_ridge_probe() -> None:
    """Plot the fixed-capacity PCA--Ridge leave-one-rho-out probe."""
    probe = plots.pd.read_csv(plots.RESULTS / "abtab2_layerwise_probe.csv")
    fig, ax = plots.plt.subplots(figsize=(10.5, 8.5))
    specs = [
        ("S_AA", r"$S_{AA}$", plots.COLORS["blue"]),
        ("S_BB", r"$S_{BB}$", plots.COLORS["orange"]),
        ("S_AB", r"$S_{AB}$", plots.COLORS["teal"]),
        ("q_star", r"$k^*$", plots.COLORS["purple"]),
        ("chiN_star", r"$\chi N^*$", plots.COLORS["red"]),
    ]
    for concept, label, color in specs:
        frame = probe[probe.concept == concept].sort_values("layer")
        ax.plot(
            frame.layer,
            frame.R2_grouped_by_rho,
            "o-",
            color=color,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.0,
            linewidth=2.8,
            label=label,
        )
    minimum = float(probe.R2_grouped_by_rho.min())
    lower = min(-0.10, minimum - 0.08)
    ax.axhline(0.0, color=plots.COLORS["black"], linewidth=1.2, alpha=0.55, zorder=0)
    ax.set_xlabel("Representation stage")
    ax.set_ylabel(r"PCA--Ridge probe $R^2$ (LO-$\rho$)")
    ax.set_xticks(
        range(6),
        ["Input", "GINE-1", "GINE-2", "GINE-3", "GINE-4", "GINE-5"],
    )
    ax.set_ylim(lower, 1.05)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=17,
        handlelength=1.5,
    )
    plots.style_axis(ax)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    fig.subplots_adjust(left=0.17, right=0.95, bottom=0.16, top=0.82)
    plots.save_dual(fig, plots.RESULTS / "abtab2_layerwise_probe", tight=False)
    plots.plt.close(fig)


if __name__ == "__main__":
    plots.plot_rpa_components()
    plots.plot_rpa_total_deletion()
    plot_layerwise_pca_ridge_probe()
    plots.plot_path_distance()
    print(f"saved panels (a)--(d) to {figure_dir}")
