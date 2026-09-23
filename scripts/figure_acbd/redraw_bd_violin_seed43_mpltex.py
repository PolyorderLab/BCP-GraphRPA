#!/usr/bin/env python3
"""Redraw only panels b/d; panels a/c remain untouched in the master artwork."""

from __future__ import annotations

import os
import sys
from pathlib import Path



ROOT = Path(__file__).resolve().parents[2]


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from paper_style import configure_paper_style, save_dual, style_distribution_axis


TRAIN_TABLE = ROOT / "data/figure_acbd/responses/training_100k_response_data.csv"
TEST_TABLE = ROOT / "data/figure_acbd/responses/validation_seed43_2k_response_data.csv"
OOD_MANIFEST = ROOT / "data/figure_acbd/responses/ood_6to10blocks_response_manifest.csv"
OUT_STEM = ROOT / "results/figures/figure_acbd/bd_response_distributions"

# Preserve the reference palette exactly.
TRAIN_COLOR, TEST_COLOR, OOD_COLOR = "#4C78A8", "#F58518", "#54A24B"


def load_test() -> pd.DataFrame:
    frame = pd.read_csv(TEST_TABLE, usecols=["n_blocks", "log10_chiN_star"])
    frame["dataset"] = "Test"
    if len(frame) != 2000:
        raise RuntimeError(f"Expected 2,000 seed43 independent test samples, found {len(frame)}.")
    return frame


def draw(
    ax,
    frame: pd.DataFrame,
    blocks: list[int],
    palette: dict[str, str],
    split: bool,
    ylim: tuple[float, float],
) -> None:
    labels = [f"{n} blocks" for n in blocks]
    frame = frame.copy()
    frame["block_label"] = pd.Categorical(
        frame["n_blocks"].map(lambda n: f"{n} blocks"), categories=labels, ordered=True
    )
    sns.violinplot(
        data=frame, x="block_label", y="log10_chiN_star", hue="dataset",
        order=labels, hue_order=list(palette), palette=palette, split=split,
        inner="quartile", cut=0, density_norm="area", common_norm=False,
        linewidth=2.0, saturation=1, ax=ax,
    )
    ax.set(xlabel="", ylabel=r"$\log_{10}(\chi N^*)$", ylim=ylim)
    ax.set_yticks(np.arange(1.0, 3.01, 0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.set_ylabel(r"$\log_{10}(\chi N^*)$", fontsize=30)
    # Matplotlib expresses box aspect as height / width: 2 / 3 gives 3:2.
    ax.set_box_aspect(2.0 / 3.0)
    style_distribution_axis(ax)
    # Keep the frame and internal quartile lines at the 2.8 pt reference
    # stroke, while using the requested 2.0 pt violin outlines.
    for spine in ax.spines.values():
        spine.set_linewidth(2.8)
    for collection in ax.collections:
        collection.set_linewidth(2.0)
        collection.set_edgecolor("#333333")
    for line in ax.lines:
        line.set_linewidth(2.8)
    ax.tick_params(which="major", labelsize=24)
    ax.grid(False)
    if ax.get_legend() is not None:
        ax.get_legend().remove()


def save_single_panel(
    frame: pd.DataFrame,
    blocks: list[int],
    palette: dict[str, str],
    split: bool,
    ylim: tuple[float, float],
    legend_handles: list[Patch],
    stem: Path,
) -> None:
    """Export a standalone panel for lossless a--d composition."""
    fig, ax = plt.subplots(figsize=(12.0, 8.1), facecolor="white")
    draw(ax, frame, blocks, palette, split=split, ylim=ylim)
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=24,
              ncol=len(legend_handles), handlelength=1.8, handleheight=1.0,
              columnspacing=1.0)
    fig.subplots_adjust(left=0.10, right=0.965, bottom=0.13, top=0.96)
    fig.text(0.5325, 0.045, "Number of blocks", ha="center", va="center", fontsize=30)
    save_dual(fig, stem, tight=False)
    plt.close(fig)


def main() -> None:
    configure_paper_style()
    train = pd.read_csv(TRAIN_TABLE, usecols=["n_blocks", "log10_chiN_star"])
    train["dataset"] = "Train"
    test = load_test()
    ood = pd.read_csv(OOD_MANIFEST, usecols=["block_count", "log10_chiN_star"])
    ood = ood.rename(columns={"block_count": "n_blocks"})
    ood["dataset"] = "OOD test"

    # Two stacked 12 x 8-inch reference-sized panels.  These margins give
    # each axes box approximately 10.4 x 6.9 inches (width:height = 3:2),
    # matching the supplied 7185 x 4813 px reference at 600 dpi.
    fig, (ax_b, ax_d) = plt.subplots(2, 1, figsize=(12.0, 16.2), facecolor="white")
    draw(ax_b, pd.concat([train, test], ignore_index=True), [2, 3, 4, 5],
         {"Train": TRAIN_COLOR, "Test": TEST_COLOR}, split=True,
         ylim=(0.875, 3.25))
    draw(ax_d, ood, [6, 7, 8, 9, 10], {"OOD test": OOD_COLOR}, split=False,
         ylim=(0.875, 3.125))
    ax_b.legend(
        handles=[Patch(facecolor=TRAIN_COLOR, edgecolor="none", label="Train"),
                 Patch(facecolor=TEST_COLOR, edgecolor="none", label="Test")],
        loc="upper left", ncol=2, frameon=False, fontsize=24,
        handlelength=1.8, handleheight=1.0, columnspacing=1.0,
    )
    ax_d.legend(
        handles=[Patch(facecolor=OOD_COLOR, edgecolor="none", label="OOD test")],
        loc="upper left", frameon=False, fontsize=24,
        handlelength=1.8, handleheight=1.0,
    )
    fig.subplots_adjust(left=0.10, right=0.965, bottom=0.10, top=0.98, hspace=0.06)
    # Centre the shared label on the actual axes boxes, not on the full canvas.
    fig.canvas.draw()
    lower_box = ax_d.get_position()
    axes_center = 0.5 * (lower_box.x0 + lower_box.x1)
    fig.text(axes_center, 0.035, "Number of blocks", ha="center", va="center", fontsize=30)
    save_dual(fig, OUT_STEM, tight=False)
    plt.close(fig)
    save_single_panel(
        pd.concat([train, test], ignore_index=True), [2, 3, 4, 5],
        {"Train": TRAIN_COLOR, "Test": TEST_COLOR}, True, (0.875, 3.25),
        [Patch(facecolor=TRAIN_COLOR, edgecolor="none", label="Train"),
         Patch(facecolor=TEST_COLOR, edgecolor="none", label="Test")],
        ROOT / "results/figures/figure_acbd/panel_b_response_distribution",
    )
    save_single_panel(
        ood, [6, 7, 8, 9, 10], {"OOD test": OOD_COLOR}, False, (0.875, 3.125),
        [Patch(facecolor=OOD_COLOR, edgecolor="none", label="OOD test")],
        ROOT / "results/figures/figure_acbd/panel_d_response_distribution",
    )
    print(f"Saved {OUT_STEM}.png/.svg")


if __name__ == "__main__":
    main()
