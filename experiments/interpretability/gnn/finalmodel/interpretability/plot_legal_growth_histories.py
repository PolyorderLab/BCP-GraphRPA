#!/usr/bin/env python3
"""Plot matched-path results averaged over all legal graph-growth histories."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "figure" / "results_legal_growth_histories"
OUT = HERE.parent / "figure" / "c2_model" / "interpretability_ablation"

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 24,
    "axes.labelsize": 30,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "axes.linewidth": 2.5,
    "xtick.major.width": 2.5,
    "ytick.major.width": 2.5,
    "xtick.major.size": 10,
    "ytick.major.size": 10,
})


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(RESULTS / "matched_path_legal_history_pairs.csv")
    stability = pd.read_csv(RESULTS / "matched_path_legal_history_stability.csv")

    pooled = [spearmanr(pairs["concept"], pairs[f"embedding_L{i}"]).statistic for i in range(1, 6)]
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    colors = ["#4c92c3"] * 4 + ["#df5a5a"]
    bars = ax.barh(range(5), pooled, color=colors, alpha=0.9, edgecolor="#222222")
    ax.set_yticks(range(5), [f"GINE-{i}" for i in range(1, 6)])
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"Embedding--RPA path Spearman $\rho_s$")
    ax.set_ylabel("Representation layer")
    ax.tick_params(direction="in", top=True, right=True)
    for bar, value in zip(bars, pooled):
        ax.text(value + 0.018, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    fig.subplots_adjust(left=0.22, right=0.95, bottom=0.17, top=0.95)
    save(fig, "abtab2_matched_path")

    group_map = stability[["group", "f_A", "rho"]].drop_duplicates().set_index("group")
    cell_rho = pairs.groupby("group").apply(
        lambda g: spearmanr(g["concept"], g["embedding_L5"]).statistic,
        include_groups=False,
    )
    f_values = sorted(group_map["f_A"].unique())
    rho_values = sorted(group_map["rho"].unique())
    matrix = np.empty((len(f_values), len(rho_values)))
    for group, value in cell_rho.items():
        row = f_values.index(group_map.loc[group, "f_A"])
        col = rho_values.index(group_map.loc[group, "rho"])
        matrix[row, col] = value

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(rho_values)), [f"{x:g}" for x in rho_values])
    ax.set_yticks(range(len(f_values)), [f"{x:.1f}" for x in f_values])
    ax.set_xlabel(r"Arm-ratio parameter $\rho$")
    ax.set_ylabel(r"A fraction $f_A$")
    ax.tick_params(direction="in", top=True, right=True)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] >= 0.55 else "#222222"
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color=color, fontsize=21)
    cbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.035)
    cbar.set_label(r"Spearman $\rho_s$")
    cbar.ax.tick_params(direction="in", width=2.5, length=8)
    fig.subplots_adjust(left=0.17, right=0.91, bottom=0.16, top=0.95)
    save(fig, "abtab2_matched_path_grid")


if __name__ == "__main__":
    main()
