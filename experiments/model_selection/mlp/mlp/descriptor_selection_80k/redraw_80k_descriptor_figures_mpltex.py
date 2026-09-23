#!/usr/bin/env python3
"""Redraw descriptor-screening Figures 01--05 from the isolated 80k fold.

This is a read-only plotting utility: all inputs are the audited CSV files in
``analysis_80k_trainfold``.  It deliberately does not read ``analysis_100k``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tapd-80k-descriptor")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis_80k_trainfold"
STYLE_DIR = Path("/home/zxy/.codex/skills/mpltex-skill/scripts")
sys.path.insert(0, str(STYLE_DIR))
from paper_style import COLORS, configure_paper_style, save_dual, style_axis  # noqa: E402


def finish(fig: plt.Figure, stem: Path) -> None:
    """Export the mandatory PNG/SVG pair and release the canvas."""
    save_dual(fig, stem)
    plt.close(fig)


def heatmap(path: Path, stem: str, label: str) -> None:
    corr = pd.read_csv(path, index_col=0)
    values = corr.to_numpy(dtype=float)
    values[np.triu_indices_from(values, k=1)] = np.nan
    cmap = LinearSegmentedColormap.from_list(
        "tapd_diverging", [COLORS["blue"], "#FFFFFF", COLORS["red"]]
    ).copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(18.5, 17.0), facecolor="white")
    image = ax.imshow(values, vmin=-1, vmax=1, cmap=cmap, aspect="equal", interpolation="none")
    ticks = np.arange(len(corr))
    ax.set_xticks(ticks, corr.columns, rotation=90)
    ax.set_yticks(ticks, corr.index)
    ax.set_title(f"Pairwise descriptor {label} correlation (80k fitting fold)", fontsize=30, pad=18)
    style_axis(ax)
    ax.minorticks_off()
    ax.tick_params(which="major", labelsize=19, length=8, width=2.4)
    cbar = fig.colorbar(image, ax=ax, fraction=0.038, pad=0.025)
    cbar.set_label(f"{label} correlation", fontsize=30, labelpad=18)
    style_axis(cbar.ax)
    cbar.ax.tick_params(which="major", labelsize=24, length=9, width=2.2)
    fig.tight_layout()
    finish(fig, ANALYSIS_DIR / "figures" / stem)


def target_correlations() -> None:
    raw = pd.read_csv(ANALYSIS_DIR / "descriptor_chiN_star_correlation_train.csv")
    log = pd.read_csv(ANALYSIS_DIR / "descriptor_log10_chiN_star_correlation_train.csv")
    names = log.nlargest(15, "abs_spearman_r")["descriptor"].tolist()
    raw = raw.set_index("descriptor").loc[names].iloc[::-1]
    log = log.set_index("descriptor").loc[names].iloc[::-1]
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(18.0, 10.0), sharey=True, facecolor="white")
    for ax, frame, title in zip(axes, (raw, log), (r"Raw $\chi N^*$", r"$\log_{10}(\chi N^*)$")):
        ax.barh(y - 0.19, frame["pearson_r"], height=0.36, color=COLORS["blue"], label="Pearson")
        ax.barh(y + 0.19, frame["spearman_r"], height=0.36, color=COLORS["orange"], label="Spearman")
        ax.axvline(0, color=COLORS["black"], lw=2.2)
        ax.set(xlabel="Correlation coefficient", title=title)
        style_axis(ax)
    axes[0].set_yticks(y, names)
    axes[0].legend(loc="lower right")
    fig.suptitle("Top descriptor–target correlations (80k fitting fold)", fontsize=30, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.4)
    finish(fig, ANALYSIS_DIR / "figures" / "02_descriptor_target_correlations")


def pruning() -> None:
    corr = pd.read_csv(ANALYSIS_DIR / "correlation_pruning_train.csv").sort_values(
        "abs_spearman_between_descriptors"
    )
    vif = pd.read_csv(ANALYSIS_DIR / "vif_pruning_train.csv").sort_values("VIF_at_removal")
    fig, axes = plt.subplots(1, 2, figsize=(19.0, 11.0), facecolor="white")
    axes[0].barh(corr["removed"], corr["abs_spearman_between_descriptors"], color=COLORS["orange"])
    axes[0].axvline(0.95, color=COLORS["black"], ls="--", label="Threshold = 0.95")
    axes[0].set(xlabel=r"$|$Spearman $r|$ with retained descriptor", title="Correlation pruning")
    axes[0].legend(loc="lower right")
    axes[1].barh(vif["removed"], vif["VIF_at_removal"], color=COLORS["purple"])
    axes[1].axvline(10, color=COLORS["black"], ls="--", label="Threshold = 10")
    axes[1].set(xscale="log", xlabel="Variance inflation factor", title="Iterative VIF pruning")
    axes[1].legend(loc="lower right")
    for ax in axes:
        style_axis(ax)
    for ax in axes:
        ax.tick_params(axis="y", labelsize=20)
    fig.tight_layout(w_pad=2.2)
    finish(fig, ANALYSIS_DIR / "figures" / "03_correlation_and_vif_pruning")


def pca() -> None:
    frame = pd.read_csv(ANALYSIS_DIR / "pca_explained_variance_train.csv")
    fig, ax = plt.subplots(figsize=(11.5, 8.5), facecolor="white")
    ax.bar(frame["component"], frame["explained_variance_ratio"], color=COLORS["blue"], label="Individual")
    ax.plot(frame["component"], frame["cumulative_explained_variance_ratio"], color=COLORS["orange"], marker="o", label="Cumulative")
    ax.axhline(0.95, color=COLORS["black"], ls="--", label="95% threshold")
    ax.axvline(7, color=COLORS["black"], ls=":", label="7 components")
    ax.set(xlim=(0.35, len(frame) + 0.65), ylim=(0, 1.05), xlabel="Principal component", ylabel="Explained variance ratio",
           title="PCA of the standardized 35-descriptor matrix (80k fitting fold)")
    ax.legend(loc="center right")
    style_axis(ax)
    fig.tight_layout()
    finish(fig, ANALYSIS_DIR / "figures" / "04_pca_explained_variance")


def automatic_selection() -> None:
    mi = pd.read_csv(ANALYSIS_DIR / "mutual_information_train.csv").set_index("descriptor")
    lasso = pd.read_csv(ANALYSIS_DIR / "lasso_coefficients_train.csv").set_index("descriptor")
    trees = pd.read_csv(ANALYSIS_DIR / "extra_trees_importance_train.csv").set_index("descriptor")
    summary = pd.read_csv(ANALYSIS_DIR / "feature_selection_summary_train.csv")
    names = summary["descriptor"].tolist()
    frame = pd.DataFrame({
        "Mutual\ninformation": mi.loc[names, "mutual_information"],
        "|Lasso|\ncoefficient": lasso.loc[names, "abs_coefficient"],
        "ExtraTrees\nimportance": trees.loc[names, "extra_trees_importance"],
    })
    normalized = (frame - frame.min()) / (frame.max() - frame.min()).replace(0, 1)
    normalized = normalized.loc[frame.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(13.0, 11.5), facecolor="white")
    selection_cmap = LinearSegmentedColormap.from_list(
        "tapd_selection", [COLORS["blue"], "#FFFFFF", COLORS["red"]]
    )
    image = ax.imshow(
        normalized.to_numpy(), cmap=selection_cmap, vmin=0, vmax=1,
        aspect="auto", interpolation="none",
    )
    ax.set_xticks(np.arange(len(normalized.columns)), normalized.columns)
    ax.set_yticks(np.arange(len(normalized.index)), normalized.index)
    for row in range(normalized.shape[0]):
        for col in range(normalized.shape[1]):
            value = normalized.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=19,
                    color="white" if value > 0.58 else COLORS["black"])
    ax.set_title("Automatic feature-selection signals after correlation/VIF screening", fontsize=31, pad=20)
    style_axis(ax)
    ax.minorticks_off()
    ax.tick_params(which="major", labelsize=22, length=10, width=2.4)
    cbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Within-method min–max normalized score", fontsize=28, labelpad=18)
    style_axis(cbar.ax)
    cbar.ax.tick_params(which="major", labelsize=22, length=9, width=2.2)
    fig.tight_layout()
    finish(fig, ANALYSIS_DIR / "figures" / "05_automatic_feature_selection")


def main() -> None:
    configure_paper_style()
    (ANALYSIS_DIR / "figures").mkdir(exist_ok=True)
    heatmap(ANALYSIS_DIR / "descriptor_pearson_correlation_train.csv", "01_descriptor_pearson_heatmap", "Pearson")
    heatmap(ANALYSIS_DIR / "descriptor_spearman_correlation_train.csv", "01_descriptor_spearman_heatmap", "Spearman")
    target_correlations()
    pruning()
    pca()
    automatic_selection()
    print(f"Wrote Figures 01--05 to {ANALYSIS_DIR / 'figures'}")


if __name__ == "__main__":
    main()
