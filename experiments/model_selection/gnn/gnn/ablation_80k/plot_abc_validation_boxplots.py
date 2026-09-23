#!/usr/bin/env python3
"""Plot the frozen A/B/C GNN validation ablations as three box-plot panels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter


SKILL_SCRIPTS = Path("/home/zxy/.codex/skills/mpltex-skill/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from paper_style import (  # noqa: E402
    COLORS,
    configure_paper_style,
    save_dual,
    style_distribution_axis,
)


configure_paper_style()

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE / "runs_100k_fixed80_20_validation_only"
SUMMARY = RUN_ROOT / "summary" / "ablation_validation_summary.csv"
OUTPUT_STEM = RUN_ROOT / "figures" / "gnn_abc_validation_mape_boxplots"

PANELS = (
    {
        "title": "(a) Edge inputs",
        "keys": ("S01", "A2", "A3"),
        "labels": ("A1\nFull", "A2\n$f$ only", "A3\nA/B only"),
        "log_y": True,
        "ylim": (0.50, 95.0),
        "n_position": (0.96, 0.05, "right", "bottom"),
    },
    {
        "title": "(b) Graph construction",
        "keys": ("S01", "B2", "B3", "B4", "B5"),
        "labels": (
            "B1\nVN",
            "B2\nNo VN",
            "B3\nLine",
            "B4\nBipart.",
            "B5\nSelf-loop",
        ),
        "log_y": True,
        "ylim": (0.50, 60.0),
        "n_position": (0.96, 0.95, "right", "top"),
    },
    {
        "title": "(c) Node inputs",
        "keys": ("S01", "C2", "C3"),
        "labels": ("C1\nEngineered", "C2\nLearnable", "C3\nDegree +\nneighbor"),
        "log_y": False,
        "ylim": (0.55, 1.55),
        "n_position": (0.04, 0.95, "left", "top"),
    },
)


def _load_scores(path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    scores: dict[str, np.ndarray] = {}
    for row in frame.itertuples(index=False):
        values = np.asarray(json.loads(row.validation_mape_per_seed_percent), dtype=float)
        if values.size != 5 or not np.all(np.isfinite(values)):
            raise RuntimeError(f"Expected five finite seed scores for {row.condition_key}")
        aliases = str(row.aliases).split(";")
        canonical = "S01" if "S01" in str(row.condition_key) else aliases[0]
        scores[canonical] = values
    required = {key for panel in PANELS for key in panel["keys"]}
    missing = required.difference(scores)
    if missing:
        raise RuntimeError(f"Missing ablation conditions: {sorted(missing)}")
    return scores


def _plot_panel(ax, scores: dict[str, np.ndarray], panel: dict) -> None:
    values = [scores[key] for key in panel["keys"]]
    positions = np.arange(1, len(values) + 1, dtype=float)
    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        boxprops={"edgecolor": COLORS["black"], "linewidth": 1.6},
        medianprops={"color": COLORS["black"], "linewidth": 2.2},
        whiskerprops={"color": COLORS["black"], "linewidth": 1.6},
        capprops={"color": COLORS["black"], "linewidth": 1.6},
    )
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor(COLORS["orange"] if index == 0 else COLORS["blue"])
        patch.set_alpha(0.72)

    # Deterministic offsets expose every seed without implying another variable.
    offsets = np.linspace(-0.12, 0.12, 5)
    for position, seed_scores in zip(positions, values):
        ax.scatter(
            position + offsets,
            seed_scores,
            s=52,
            color=COLORS["black"],
            edgecolors="#FFFFFF",
            linewidths=0.8,
            zorder=4,
        )

    style_distribution_axis(ax)
    if panel["log_y"]:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=5))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.text(
            0.04,
            0.95,
            "log scale",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=18,
            color="0.25",
        )

    ax.set_xlim(0.48, len(values) + 0.52)
    ax.set_ylim(*panel["ylim"])
    ax.set_xticks(positions)
    x_label_size = 16 if len(values) == 5 else 18
    ax.set_xticklabels(panel["labels"], fontsize=x_label_size, linespacing=0.92)
    ax.set_title(panel["title"], fontsize=32, pad=12)
    ax.set_ylabel("Validation MAPE (%)", fontsize=27)
    n_x, n_y, n_ha, n_va = panel["n_position"]
    ax.text(
        n_x,
        n_y,
        "$n=5$ seeds",
        transform=ax.transAxes,
        ha=n_ha,
        va=n_va,
        fontsize=18,
        color="0.25",
    )
    ax.tick_params(axis="y", which="major", labelsize=21)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--output-stem", type=Path, default=OUTPUT_STEM)
    args = parser.parse_args()

    scores = _load_scores(args.summary)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.9), facecolor="white")
    for ax, panel in zip(axes, PANELS):
        _plot_panel(ax, scores, panel)

    fig.subplots_adjust(left=0.075, right=0.988, bottom=0.205, top=0.87, wspace=0.36)
    save_dual(fig, args.output_stem, tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
