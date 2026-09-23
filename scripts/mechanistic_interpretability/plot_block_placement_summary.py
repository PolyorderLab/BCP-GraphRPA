#!/usr/bin/env python3
"""Plot the history-averaged block-placement correlation by GINE layer."""
from __future__ import annotations

import json
from pathlib import Path

from paper_style import COLORS, configure_paper_style, save_dual, style_axis  # noqa: E402

configure_paper_style()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


CODE_ROOT = Path(__file__).resolve().parents[2]
RESULTS = CODE_ROOT / "data/mechanistic_interpretability/legal_histories"
OUTPUT = CODE_ROOT / "results/figures/mechanistic_interpretability/block_placement_history_averaged"


def main() -> None:
    payload = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["summary"]).query("layer > 0").sort_values("layer")
    values = frame["spearman_mean_embedding_vs_RPA_path"].to_numpy(dtype=float)
    positions = np.arange(len(frame))
    colors = [COLORS["blue"]] * (len(frame) - 1) + [COLORS["red"]]

    fig, ax = plt.subplots(figsize=(10.5, 8.5), facecolor="white")
    bars = ax.barh(
        positions,
        values,
        height=0.60,
        color=colors,
        alpha=0.90,
        edgecolor=COLORS["black"],
        linewidth=0.8,
    )
    ax.set(
        yticks=positions,
        yticklabels=[f"GINE-{layer}" for layer in frame["layer"]],
        xlabel=r"History-averaged path Spearman $\rho_s$",
        ylabel="GNN layer",
        xlim=(0.0, 1.00),
        ylim=(-0.65, len(frame) - 0.35),
        xticks=np.arange(0.0, 1.01, 0.2),
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=21,
            color=COLORS["black"],
        )
    style_axis(ax)
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    fig.subplots_adjust(left=0.25, right=0.95, bottom=0.16, top=0.96)
    png, svg = save_dual(fig, OUTPUT, tight=False)
    plt.close(fig)
    print(f"Saved {png}\nSaved {svg}")


if __name__ == "__main__":
    main()
